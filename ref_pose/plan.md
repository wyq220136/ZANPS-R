# 模块功能
接收reconstruction输出的mesh，用在foundationpose基础上改进的方法，实现part的pose estimation

# 解决的问题
1. 单视角信息的局限性，只能用单帧RGBD信息做pose estimation
2. 单视角重建模型的失真性，存在不可信几何结构

# 要达到的目标
1. 解决上面的两个问题，pose estimation效果达到和foundationpose估计rigid object差不多的效果
2. 整个pipeline的叙事完整自洽，且有说服力，有理有据解决问题。

# 改进计划
## 1. Local hypothesis generation + correspondence-guided refinement + reranking

### 做什么
当前 ref_pose 基本是一个 init pose 进入 FoundationPose refiner。这个方案改成级联流程：先利用 FoundationPose-like 的局部球面采样，在输入 init pose 的小邻域内生成多个 pose candidates；然后把这些 candidates 放入 One2Any-like reference-query correspondence 阶段做中层校正；再回到 FoundationPose render-based refinement 做精修；最后用一致性分数选择最可信 pose。

### 输入
- reconstruction 输出的 `model.obj`
- `raw_pose.txt` 或 match 阶段的 `coarse_pose/refined_pose`
- query RGB-D、query part mask、camera K
- 可选：`reference_points.npy`、connector vector、edge gate 信息

### 输出
- 每个 part 的 best refined pose
- 每个候选的 local sampled pose、correspondence-corrected pose、FoundationPose refined pose 和 score log
- 可视化：候选 pose overlay、mask/depth residual、最终选择原因

### 提升效果的依据
单视角 reference 信息不足，coarse init 容易落入错误局部最优。局部球面采样把单个 init pose 扩展为多个可能解释；reference-query correspondence 阶段利用真实 reference evidence 对候选做中层校正；FoundationPose refinement 再利用 mesh render 做最终精修；最后 reranking 避免某个阶段的错误直接决定最终结果。

### 具体实现
1. 修改 `ref_pose/foundationpose.py`
   - 增加 local pose candidate sampling 接口。
   - 当上游只给一个 init pose 时，围绕该 init pose 的局部旋转邻域生成多个 candidates。
   - 保留原始 FoundationPose register 的默认行为，避免破坏旧调用。

2. 修改 `ref_pose/reference_evidence.py`
   - 增加 lightweight reference-query correspondence correction。
   - 对每个 candidate，把 reference-visible points 投到 query view，利用 query depth/mask 估计一个小范围 camera-space correction。
   - 输出 correspondence-corrected candidate 和可解释 correspondence 日志。

3. 修改 `ref_pose/estimater.py`
   - `pose_single_estimation` 增加参数：
     - `init_pose_candidate_overrides: dict[int, list[np.ndarray]] | None`
     - `pose_rerank: bool = True`
     - `use_fp_local_candidates: bool = True`
     - `use_reference_correspondence: bool = True`
   - 对每个 part：
     - 如果上游没有给 Top-K，就用 FoundationPose 局部采样从单个 init pose 生成 candidates
     - 对每个 candidate 先做 reference-query correspondence 校正
     - 再调用 `est.register(..., init_pose=correspondence_pose)` 做 FoundationPose refinement
     - 保存 refined pose
     - 调用 consistency score 模块打分
   - 选择 best pose 写入 `ob_in_cam2/.../pose_xxxx.txt`。

4. 可选扩展上游
   - 如果后续允许修改 ref_pose 之外的模块，可以让 `estimate_frame_init_poses_fast` 直接返回外部 Top-K。
   - 当前实现不依赖上游 Top-K，保证上游只给单 pose 时仍可运行。

### 借鉴来源
- Any6D：multi-hypothesis initialization、render-and-compare verification、coarse-to-fine pose refinement。
- One2Any：reference-query correspondence 用于中层 pose 校正。

### 借鉴来源的实现方式（方法级）
输入是单视角 reference 形成的 object/part prior、query RGB-D、query mask 和相机内参。第一步先围绕上游 init pose 做局部球面采样，生成多个 pose hypotheses，而不是只保留一个初值；第二步用 reference-query correspondence 对每个 hypothesis 做中层校正，使候选先满足 reference evidence 与 query observation 的几何对应；第三步再对每个 corrected hypothesis 做 FoundationPose render-based refinement；第四步把每个 refined candidate 重新验证，用 mask overlap、depth agreement、reference consistency 和 validity confidence 进行打分；最后输出得分最高的 refined pose，以及每个候选的可解释分数。这个流程借鉴 Any6D 中“先生成多个可能解释，再用观测一致性验证”的思想，也借鉴 One2Any 中 reference-query correspondence 先于最终 pose refinement 的思想。

## 2. Overlap / reliability-aware validity mask

### 做什么
把当前 refiner 中已有的 `validity_mask` 明确升级成 reliability-aware refinement 机制：refiner 不平均相信整个 reconstructed mesh，而是在 query 条件下预测哪些 rendered regions 是可信且对 pose update 有用的区域。

### 输入
- rendered mesh crop：RGB、XYZ/depth
- query crop：RGB、XYZ/depth
- 当前 pose candidate
- reconstruction mesh
- query part mask
- 可选：reference-visible region mask

### 输出
- `geom_validity`
- `pose_utility`
- `validity_mask = geom_validity * pose_utility`
- 加权后的 pose delta
- mask 可视化，用于解释 refiner 依赖了哪些区域

### 提升效果的依据
reconstruction mesh 来自单视角生成补全，部分区域可能是 hallucination。直接使用全 mesh render 进行 pose refinement，会让错误几何误导 SE(3) 更新。validity mask 让 refiner 聚焦 query 中可见、与观测一致、对 pose 有贡献的区域。

### 具体实现
1. 现有代码基础
   - `ref_pose/learning/models/refine_network.py` 已有：
     - `geom_head`
     - `pose_head`
     - `output['geom_validity']`
     - `output['pose_utility']`
     - `output['validity_mask']`
   - `ref_pose/learning/training/predict_pose_refine.py` 已用上一轮 `prev_validity_mask` gate rendered feature A。

2. 修改 `ref_pose/learning/training/predict_pose_refine.py`
   - 保存每次 iteration 的：
     - validity mask mean
     - geom validity mean
     - pose utility mean
     - trans/rot update magnitude
   - 增加 `get_vis` 时输出 validity heatmap。
   - 将 `last_validity_mask` 扩展为 `last_refine_trace`，包含每轮可解释信息。

3. 修改 `ref_pose/estimater.py`
   - 每个 part 保存：
     - `pose_xxxx.txt`
     - `pose_xxxx_refine_trace.json`
     - `pose_xxxx_validity.png`
   - trace 中记录 validity mask 置信度，供 reranking 使用。

4. 修改训练脚本
   - `ref_pose/learning/training/train_ddp.py`
   - `ref_pose/learning/training/train_refine_validity_mask.py`
   - 强化现有 soft validity target：
     - depth residual 小的区域权重大
     - rendered silhouette 与 query mask overlap 区域权重大
     - RGB/XYZ residual 大、query 不可见、遮挡区域权重小
   - 保持现有 backbone 冻结，只训练 mask decoder 或轻量 adapter，降低训练风险。

### 借鉴来源
- UNOPose：overlap-aware matching、visibility/reliability weighting、occlusion-aware refinement。

### 借鉴来源的实现方式（方法级）
输入是当前 pose 下的 rendered part crop、query RGB-D crop、query mask 和当前 candidate pose。第一步估计 rendered observation 与 query observation 中真正可能对应的重叠区域；第二步对这些区域赋予 reliability/visibility 权重，把被遮挡、不可见、深度不一致或疑似 hallucination 的区域降权；第三步只让高可靠、高 pose utility 的区域主导 SE(3) delta 预测；最后输出 refined pose、validity/utility mask 和每轮 refinement 的可信区域可视化。这个流程借鉴的是 UNOPose 中“不把 reference-query 的所有区域都当作可匹配区域，而是显式建模 overlap 和可靠性”的思想。

## 3. Reference evidence anchoring

### 做什么
在 ref_pose 中正式利用 reconstruction 阶段保存的 reference evidence，而不是只使用生成后的 `model.obj`。核心思想是：mesh 是 generative prior，reference evidence 是真实观测锚点。refiner 应该知道哪些几何来自真实 reference view，哪些只是补全。

### 输入
- `model.obj`
- `reference_points.npy`
- `reference_points_obj.npy`
- `reference_points_cam.npy`
- `raw_pose.txt`
- `local_to_object.txt`
- `local_to_reference_camera.txt`
- query RGB-D、mask、K

### 输出
- reference-visible region mask 或 per-vertex/per-point reliability
- reference-guided init pose candidates
- reference evidence consistency score
- 可解释日志：当前 pose 和 reference-visible evidence 的对齐质量

### 提升效果的依据
单视角重建 mesh 不完全可信，但 reference-visible points 来自真实 RGB-D observation，更可靠。把 reference evidence 纳入 ref_pose，可以避免 refiner 被 mesh 的 hallucinated region 误导，并为 init、score 和 validity mask 提供真实观测约束。

### 具体实现
1. 修改 `ref_pose/dataloader.py`
   - `SingleLoader.reinit` 中除了读取 `model.obj` 和 `raw_pose.txt`，也读取：
     - `reference_points.npy`
     - `reference_points_obj.npy`
     - `reference_points_cam.npy`
     - `local_to_object.txt`
     - `local_to_reference_camera.txt`
   - 新增方法：
     - `get_reference_points(i, frame='object' | 'camera' | 'local')`
     - `get_reference_transforms(i)`
     - `get_reference_metadata(i)`

2. 修改 `evaluation/evaluate.py`
   - `_prepare_pose_inputs_from_match_results` 当前已经复制这些文件：
     - `reference_points.npy`
     - `reference_points_obj.npy`
     - `reference_points_cam.npy`
     - `raw_pose.txt`
     - `local_to_object.txt`
     - `local_to_reference_camera.txt`
   - 增加完整性检查和日志，如果缺失则标记 `reference_evidence_available=False`。

3. 新增模块 `ref_pose/reference_evidence.py`
   - 功能：
     - 将 reference points 根据 candidate pose 投影到 query image。
     - 计算 reference-visible points 与 query depth/mask 的一致性。
     - 生成 reference-visible reliability map。
     - 计算 score：
       - projected points inside query mask ratio
       - depth residual
       - visible point inlier ratio

4. 修改 `ref_pose/estimater.py`
   - 在每个 candidate refine 前后调用 `reference_evidence.py`：
     - init 阶段：用 reference points 过滤明显错误的 candidate
     - refinement 后：把 reference consistency 纳入 reranking score
   - 保存 `reference_evidence_score.json`。

5. 可选：修改 `predict_pose_refine.py`
   - 将 reference-visible reliability map crop 到 refiner 输入尺寸。
   - 作为额外 gating mask 与 `validity_mask` 融合：
     - `final_gate = validity_mask * reference_reliability`
   - 初期建议只用于 score，不直接改网络输入，降低风险。

### 借鉴来源
- One2Any：reference object pose embedding / reference object coordinate，把 reference observation 作为 pose prior。
- UNOPose：只信任 reference-query 可能重叠、可见、可靠的区域。
- Any6D：single RGB-D reference 参与 hypothesis generation 和 verification。

### 借鉴来源的实现方式（方法级）
输入是 reconstruction mesh，以及 reference frame 中保存的真实观测证据，包括 reference points、raw reference pose、local-to-reference transform 和 query RGB-D/mask。第一步把 reference evidence 从 reconstruction 的副产物提升为 ref_pose 的显式输入，建立 reference observation 与 reconstructed mesh 之间的坐标关系；第二步根据 reference-visible points 判断 mesh 中哪些区域有真实观测支持，哪些区域主要来自生成补全；第三步把这些 reference evidence 用于 pose candidate 生成、candidate 过滤和 refined pose verification；第四步可进一步形成 reference-visible reliability map，与 validity mask 或 consistency score 融合；最后输出 reference-guided pose candidate、reference consistency score 和可靠区域提示。这个流程借鉴 One2Any 中 reference observation 作为 pose prior 的思想，也借鉴 UNOPose/Any6D 中 reference-query 可见重叠和单参考验证的处理方式。

## 4. Depth / mask / render consistency score

### 做什么
为每个 refined pose 建立显式评分函数，用 query observation 验证 pose 是否可信。这个 score 既用于 Top-K candidate reranking，也用于输出可解释性分析。

### 输入
- refined pose
- reconstruction mesh
- query RGB-D
- query mask
- camera K
- refiner trace：validity mask、pose update magnitude
- 可选：reference evidence score、edge gate score

### 输出
- `pose_score.json`
- scalar final score
- score breakdown：
  - mask IoU
  - depth residual
  - rendered visible ratio
  - outside-mask leakage
  - validity confidence
  - pose update stability
  - reference evidence consistency

### 提升效果的依据
refiner 网络输出不一定总是正确，尤其当 mesh 几何不准或 init pose 较差时。显式 consistency score 可以把最终选择建立在观测一致性上，减少 hallucinated mesh 或错误 candidate 带来的失败。

### 具体实现
1. 新增模块 `ref_pose/pose_consistency.py`
   - 输入 mesh、pose、RGB-D、mask、K。
   - 使用已有 `nvdiffrast_render` 或 FoundationPose render utility 得到 rendered depth/mask。
   - 计算：
     - `mask_iou`
     - `depth_l1_in_mask`
     - `depth_inlier_ratio`
     - `outside_mask_ratio`
     - `rendered_area_ratio`
     - `validity_mean`
     - `pose_delta_norm`
   - 输出总分：
     - `score = w_iou * mask_iou + w_depth * depth_inlier_ratio + w_validity * validity_mean - w_leak * outside_mask_ratio - w_delta * pose_delta_norm`

2. 修改 `ref_pose/estimater.py`
   - 每个 candidate refine 后调用 `pose_consistency.score_pose(...)`。
   - 将 score breakdown 写到：
     - `ob_in_cam2.../<frame>/pose_xxxx_score.json`
   - Top-K reranking 时选择 final score 最大的 candidate。

3. 修改 `ref_pose/foundationpose.py`
   - 如果已有 `ScorePredictor` 输出，则把 neural score 也加入 score breakdown。
   - 保持 rule-based consistency score 和 neural score 分开记录，便于 ablation。

4. 修改评测脚本
   - `evaluation/evaluate.py` 增加参数：
     - `--pose-rerank-score`
     - `--pose-score-weights`
   - 方便做 ablation：
     - no rerank
     - neural score only
     - depth/mask score only
     - full score

### 借鉴来源
- Any6D：render-and-compare pose verification。
- UNOPose：overlap/visibility-aware scoring，避免不可见区域影响判断。

### 借鉴来源的实现方式（方法级）
输入是 refined pose、reconstructed mesh、query RGB-D、query mask、camera K，以及可选的 validity mask 和 reference evidence score。第一步把 reconstructed mesh 按 refined pose 渲染到 query view，得到 rendered mask、rendered depth 和 visible region；第二步只在 query 可见、mask 内或 reference/query 可重叠的区域计算一致性，避免不可见区域污染评分；第三步分别计算 mask IoU、depth residual、outside-mask leakage、visible inlier ratio 和 validity confidence；第四步把这些指标组合成最终 pose score；最后输出 best pose、score breakdown 和可视化诊断。这个流程借鉴 Any6D 的 render-and-compare verification，同时吸收 UNOPose 的 overlap/visibility-aware scoring。

## 5. Reference-coordinate auxiliary branch

### 做什么
这是长期版本：在 refiner 中增加一个 auxiliary branch，预测 query crop 中每个有效像素对应的 reference/canonical coordinate，或者预测 reference-query relative pose。它不是替代 FoundationPose，而是作为级联中的 correspondence-guided mid-level alignment：先用 reference-coordinate/correspondence 分支优化局部候选，再把优化后的候选交给 FoundationPose 做 render-based final refinement。

### 输入
- rendered mesh crop
- query RGB-D crop
- reference points / canonical object coordinates
- current pose
- GT pose 或 pseudo GT pose

### 输出
- query crop 上的 `reference_coord_map` 或 `canonical_coord_map`
- correspondence confidence map
- reference-query relative pose delta 或 correspondence-corrected pose candidate
- 辅助 loss，用于训练更强 refiner

### 提升效果的依据
当前 refiner 主要从 rendered crop 和 query crop 的 feature difference 中回归 pose delta，缺少显式 correspondence。reference-coordinate branch 让网络学习“query 中这个点对应 reference/canonical part 的哪里”，可以在 FoundationPose 精修前把候选拉到更符合 reference-query 对应关系的位置。这样它承担的是中层校正和候选增强，而不是单独决定最终 pose。

### 具体实现
1. 修改 `ref_pose/learning/models/refine_network.py`
   - 在现有 `mask_decoder` 或 shared feature 后增加：
     - `coord_head`: 输出 3 通道 canonical/reference coordinate
     - `coord_conf_head`: 输出 1 通道 correspondence confidence
   - forward 输出：
     - `output['ref_coord']`
     - `output['coord_conf']`

2. 修改 dataset
   - `ref_pose/learning/datasets/sam3d_part_dataset.py`
   - 为训练样本生成 coordinate target：
     - 用 GT pose 或 pseudo GT pose 将 query depth point 反投影到 object/reference frame。
     - 将点投到 crop 尺寸，形成 `ref_coord_target`。
     - 对 query mask 内且 depth 有效的像素监督。

3. 修改训练脚本
   - `ref_pose/learning/training/train_ddp.py`
   - 增加 loss：
     - `coord_loss = SmoothL1(coord_conf * pred_coord, coord_conf * target_coord)`
     - `coord_conf_loss` 可由 target visibility/inlier 监督
   - 总 loss 中加入：
     - pose delta loss
     - validity loss
     - coordinate auxiliary loss

4. 修改 inference
   - `ref_pose/learning/training/predict_pose_refine.py`
   - 初期只保存 `ref_coord` 和 `coord_conf` 用于可视化/诊断。
   - 稳定后将 reference-coordinate branch 的输出作为 candidates 的 correspondence-correction stage：
     - `local sampled candidate -> correspondence-corrected candidate -> FoundationPose refinement`
   - coordinate confidence 也可以与 validity mask 融合：
     - `final_gate = validity_mask * coord_conf`

5. 可选：新增 coarse pose predictor
   - 根据 predicted reference coordinates 与 query depth points 做 PnP/Umeyama/RANSAC，得到 coarse pose。
   - 作为 Top-K init 的候选来源之一。

### 借鉴来源
- One2Any：Reference Object Coordinate / Reference Object Pose Embedding。
- UNOPose：reference-query correspondence confidence。

### 借鉴来源的实现方式（方法级）
输入是 reference evidence、rendered part crop、query RGB-D crop、当前 pose 和训练时可用的 GT/pseudo GT pose。第一步把 reference observation 或 reconstructed mesh 定义到一个 reference/canonical coordinate frame；第二步在 query crop 中为有效像素建立对应的 reference/canonical coordinate target；第三步让网络在预测 pose delta 的同时预测 query pixel/point 对应 reference/canonical 位置，以及该对应关系的 confidence；第四步用 coordinate consistency 和 confidence 作为辅助监督或辅助 gating；最后输出 reference coordinate map、correspondence confidence 和更稳定的 pose refinement feature。这个流程借鉴 One2Any 中显式预测 reference object coordinate 的思想，也借鉴 UNOPose 中对 reference-query correspondence 置信度建模的思想。


# 最后pipeline的叙事（怎么讲这个模块的故事）

ref_pose 模块的定位是：接收 reconstruction 模块输出的单视角重建 part mesh，在这个 mesh 不是可靠 CAD、reference 信息又不完整的条件下，完成稳健的 part pose estimation。它的核心思想不是“直接把重建 mesh 当 CAD 用 FoundationPose 跑一遍”，而是把重建 mesh 当成带不确定性的 generative part prior，再用 reference evidence、query RGB-D observation 和多阶段验证逐步约束 pose。

## 模块输入输出

### 输入

1. reconstruction 输出的 part mesh：`model.obj`。
2. reconstruction 保存的 reference evidence：`reference_points.npy`、`reference_points_obj.npy`、`reference_points_cam.npy`、`raw_pose.txt`、`local_to_object.txt`、`local_to_reference_camera.txt`。
3. 当前 query frame 的 RGB-D、part mask 和 camera intrinsic。
4. 上游给出的 init pose，例如 match 阶段的 `coarse_pose/refined_pose` 或 depth/mask alignment 得到的 pose。
5. 可选的结构信息，例如 connector vector、edge gate 或 kinematic graph consistency。

### 输出

1. 每个 part 在 query camera frame 下的 final refined pose。
2. 每个候选的中间结果：local sampled pose、correspondence-corrected pose、FoundationPose refined pose。
3. 每个候选的可解释评分：mask consistency、depth consistency、reference evidence consistency、validity confidence、pose update stability。
4. 可视化与日志，用于说明最终 pose 为什么被选中。

整体流程可以概括为：

`reconstructed part prior + reference evidence + query RGB-D/mask + init pose`
`-> local hypothesis generation`
`-> reference-query correspondence alignment`
`-> reliability-aware FoundationPose refinement`
`-> observation/reference consistency reranking`
`-> final part pose`

## 组成部分

### 1. Reconstructed Part Prior Loading

这一步读取 reconstruction 输出的 `model.obj`。在叙事里，这个 mesh 不被定义为可靠 CAD，而是定义为 single-view generative part prior。

它做的工作是给 ref_pose 提供一个可渲染、可优化、形状相对完整的几何假设。单视角 reference 只能看到 part 的局部，而 render-and-refine 方法需要一个可渲染的几何模型。重建 mesh 正好补充了 reference 不可见区域，使 pose refinement 不再只依赖稀疏可见点。

为什么有效：它缓解了单视角信息不足的问题，把“只有局部观测”扩展为“有一个完整但带不确定性的形状先验”。但它不被无条件信任，因为生成式补全可能包含错误几何。

### 2. Reference Evidence Anchoring

这一步读取 reconstruction 阶段保留下来的真实 reference evidence，例如 reference-visible points、raw reference pose 和 local-to-reference transform。

它做的工作是把 reference frame 中真实看到过的几何重新引入 ref_pose。也就是说，ref_pose 同时拥有两类信息：

1. `model.obj`：完整但可能有 hallucination 的生成先验。
2. reference evidence：不完整但来自真实 RGB-D 观测的几何锚点。

reference evidence 后续用于判断候选 pose 是否能让 reference-visible points 合理投影到 query mask 和 query depth 上，也可以形成 reference-visible reliability signal。

为什么有效：它直接针对“重建模型不可信”的问题。即使 mesh 的补全部分可能错，reference-visible points 仍然提供了真实观测支撑，使系统能够区分“真实看见过的几何”和“生成模型补出来的几何”。

### 3. Local Hypothesis Generation

这一步从上游 init pose 出发，在其局部邻域生成多个 pose candidates。即使上游只给一个 pose，ref_pose 也可以利用 FoundationPose-like local spherical sampling，在该 pose 周围采样多个局部旋转候选，并保留 init pose 本身。

它做的工作是把单一初始化扩展为候选池：

1. 如果上游给了多个 candidates，直接接收。
2. 如果上游只给一个 init pose，就在这个 pose 附近做局部球面采样。
3. 每个 candidate 只表示一个可能解释，不直接作为最终结果。

为什么有效：单视角 reference 和 query 之间天然存在视角歧义、遮挡和局部形状歧义，一个 init pose 很容易落入错误 basin。局部候选采样把“单点估计”变成“局部假设集合”，降低上游单个 pose 错误导致整体失败的风险。

### 4. Reference-query Correspondence-guided Alignment

这一步对应 One2Any-like 的中层 correspondence 优化。对每个 local pose candidate，先把 reference-visible points 按当前 candidate 投影到 query view，再利用 query depth 和 query mask 判断这些点是否有合理对应。如果存在足够可靠的对应关系，就对 candidate 做一个小范围的 camera-space correction。

它做的工作不是直接输出最终 pose，而是把粗候选先拉到更符合 reference-query 几何对应的位置：

1. 输入 local sampled candidate。
2. 投影 reference-visible points。
3. 根据 query mask 和 depth 找到可信 correspondence。
4. 估计小范围 pose correction。
5. 输出 correspondence-corrected candidate。

为什么有效：FoundationPose 的 render refinement 强依赖初值和 mesh 几何，如果候选与真实 query observation 偏差较大，或者 mesh 补全部分有误，直接 refinement 可能被带偏。reference-query correspondence 阶段利用真实 reference evidence 做中层校正，使候选在进入最终 render refinement 前已经和真实观测建立了对应关系。

### 5. Reliability-aware FoundationPose Refinement

这一步把 correspondence-corrected candidates 送入 FoundationPose-like render-and-refine 流程。每个 candidate 下，系统 render reconstructed mesh，与 query RGB-D crop 对齐，并迭代预测 SE(3) delta。

这里的关键不是普通 render matching，而是 reliability-aware refinement。refiner 使用 `geom_validity` 和 `pose_utility` 得到 `validity_mask`：

1. `geom_validity` 判断当前 rendered geometry 是否可信。
2. `pose_utility` 判断该区域是否对 pose update 有贡献。
3. `validity_mask` 控制哪些区域真正参与 pose refinement。

为什么有效：重建 mesh 可能有 hallucinated region，如果全区域平均参与 refinement，错误补全会误导 pose update。validity-aware refinement 让模型更多依赖 query 中可见、depth/mask 一致、对 pose 有判别力的区域，从而降低不可信几何的影响。

### 6. Observation and Reference Consistency Reranking

这一步对所有 refined candidates 进行统一打分和重排序。最终 pose 不是由某一个阶段直接决定，而是由 query observation 和 reference evidence 共同裁决。

评分项包括：

1. rendered mask 与 query mask 的 overlap。
2. rendered depth 与 query depth 的一致性。
3. rendered geometry 是否泄漏到 part mask 外。
4. validity mask 的平均置信度。
5. pose update 是否稳定。
6. reference-visible points 投影到 query 后是否落在 mask 内并匹配 depth。
7. 可选的 connector 或 kinematic consistency。

为什么有效：多阶段 pipeline 中任何一个阶段都可能失败。local sampling 可能产生错误候选，correspondence correction 可能被局部噪声影响，FoundationPose refinement 可能被错误 mesh 区域带偏。最终 reranking 用真实 query RGB-D/mask 和 reference evidence 做统一验证，能筛掉与观测不一致的候选。

## 为什么整体有效

这套 pipeline 的关键是让不同信息承担不同角色：

1. reconstruction mesh 提供完整形状先验，但不被当成可靠 CAD。
2. reference evidence 提供真实观测锚点，用来约束生成式几何。
3. local hypothesis generation 处理单 init pose 的不确定性。
4. correspondence-guided alignment 在 FoundationPose 前做中层校正。
5. reliability-aware refinement 在像素/区域层面降低错误几何影响。
6. consistency reranking 在 pose 层面选择最符合真实观测的结果。

对于“单视角信息不足”，它的解决方式是：不要求单个 reference/init pose 一步到位，而是通过重建先验补充形状，通过局部多候选覆盖歧义，通过 reference-query correspondence 和 query observation 逐步收窄到合理 pose。

对于“单视角重建模型不可信”，它的解决方式是：不直接相信完整 mesh，而是用 reference evidence 标定真实可见几何，用 validity mask 降低不可靠区域权重，用 depth/mask/reference consistency 在最后验证每个候选。

## 最终一句话叙事

ref_pose 将单视角重建 mesh 视为带不确定性的 generative part prior，先围绕输入 init pose 生成局部 pose hypotheses，再用 reference-query correspondence 做中层校正，随后通过 reliability-aware FoundationPose refinement 完成精细对齐，最后用 depth/mask/reference consistency 选择最终 pose，从而在单视角信息不足且重建几何可能失真的条件下实现稳健、可解释的 part pose estimation。


# 修改日志

## 2026-07-13

### 级联式第一条与第五条更新

1. `ref_pose/foundationpose.py`
   - 增加局部候选采样能力。
   - 当上游只提供一个 init pose 时，可以围绕该 pose 的局部球面邻域生成多个 pose candidates。
   - 该阶段只负责候选生成，不直接决定最终 pose。

2. `ref_pose/reference_evidence.py`
   - 增加 reference-query correspondence 的轻量几何校正功能。
   - 每个局部 pose candidate 会先经过 reference-visible points 与 query depth/mask 的对应检查，并得到一个小范围校正后的 candidate。
   - 该功能作为 One2Any-like correspondence stage 的可运行几何版本，用于后续 learned reference-coordinate branch 接入前的级联验证。

3. `ref_pose/estimater.py`
   - 将第一条和第五条整合为级联流程：
     - 上游单 init pose
     - FoundationPose-like local spherical sampling 生成 candidates
     - reference-query correspondence 校正 candidates
     - FoundationPose render-based refinement
     - depth/mask/reference/validity consistency reranking
   - 输出日志中增加 correspondence-corrected pose 和 correspondence stage 的解释信息。

4. `ref_pose/plan.md`
   - 更新第一条改进计划，从单纯 Top-K reranking 改为 local hypothesis generation + correspondence-guided refinement + reranking。
   - 更新第五条改进计划，将 reference-coordinate auxiliary branch 明确定位为 FoundationPose 前的中层 correspondence-guided alignment，而不是独立替代 FoundationPose 的链路。

1. `ref_pose/reference_evidence.py`
   - 新增 reference evidence 加载与评分功能。
   - 支持读取 reconstruction 阶段保存的 reference points、raw pose、local-to-object、local-to-reference-camera 等信息。
   - 支持把 reference-visible points 投影到 query view，并输出 reference evidence consistency score。
   - 支持生成 reference-visible reliability map，作为后续 reliability-aware refinement 或 pose verification 的观测锚点。

2. `ref_pose/pose_consistency.py`
   - 新增 pose consistency scoring 功能。
   - 支持用 query depth、query mask、reconstructed mesh、refined pose 和 reference evidence 计算可解释 pose score。
   - 输出 mask overlap、depth consistency、outside-mask leakage、validity confidence、reference consistency 和 pose delta stability 等分项指标。
   - 用于 Top-K candidate reranking 和最终 refined pose 的可解释验证。

3. `ref_pose/dataloader.py`
   - `SingleLoader` 增加 reference evidence 读取能力。
   - 每个 part model 除了读取 `model.obj` 和 `raw_pose.txt`，也会同步整理 reference points 与 reference transforms。
   - 新增 reference evidence 查询接口，供 ref_pose estimation、candidate verification 和 reranking 使用。
   - 保持原有训练和推理 I/O 兼容，不要求重新重建数据。

4. `ref_pose/estimater.py`
   - `pose_single_estimation` 增加 Top-K init pose candidate 输入能力。
   - 支持对多个 init candidates 分别执行 FoundationPose refinement。
   - 支持对 refined candidates 执行 observation/reference consistency reranking，选择最终 pose。
   - 每个 part 输出 pose 的同时，额外保存 candidate score 和 refinement trace，增强结果可解释性。
   - 旧的单 init pose 调用方式保持兼容。

5. `ref_pose/foundationpose.py`
   - `register` 增加可选 candidate 返回能力。
   - 默认行为仍返回 best pose；打开候选返回时，可同步输出 FoundationPose 内部 scorer 排序后的 refined candidates。
   - 该功能用于后续 multi-hypothesis refinement 与 reranking，不改变现有调用路径。

6. `ref_pose/learning/training/predict_pose_refine.py`
   - PoseRefinePredictor 增加 refinement trace 记录。
   - 每轮 refinement 记录 validity、geom validity、pose utility 和 pose update magnitude。
   - 输出 final validity confidence，供 pose consistency score 和可解释日志使用。
   - 不改变现有 refiner 输入输出主接口。

7. `ref_pose/learning/models/refine_network.py`
   - 增加默认关闭的 reference-coordinate auxiliary branch。
   - 当配置启用时，网络可额外输出 reference/canonical coordinate map 和 correspondence confidence。
   - 默认关闭时保持旧 checkpoint、旧训练流程和现有 I/O 兼容。

8. `ref_pose/learning/training/train_ddp.py`
   - 增加 reference-coordinate auxiliary branch 的训练开关。
   - 默认关闭，确保 `ref_pose/learning/training/run_train_ddp_workflow.sh train` 可以继续按现有 validity-mask 训练流程运行。
   - 当启用 auxiliary branch 时，训练脚本会把新增分支纳入可训练参数与 checkpoint 保存。

9. 本轮实现与三个借鉴方法的对应关系
   - Any6D：实现 multi-hypothesis refinement 和 render/depth/mask consistency verification 的 ref_pose 内部版本。
   - UNOPose：实现 reliability/overlap-aware validity trace，并把可靠区域置信度纳入 pose verification。
   - One2Any：实现 reference evidence anchoring，并预留 reference-coordinate auxiliary branch。
