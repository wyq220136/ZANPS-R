# 改进计划

## CAPNet 可借鉴模块与落地方案

### 1. Object-conditioned part feature lifting：在完整物体上下文中提取 part 特征

**要实现的功能：**  
当前 reconstruction 基本是按 part 独立处理：每个 part 根据自己的 mask、depth、pose 做参考帧重建、TSDF 融合和 DLMesh 优化。可以借鉴 CAPNet 的做法，先在完整 object ROI 上提取 RGB-D 特征，再把这些特征按照像素位置 lift 到每个 part 的 3D 点上。这样每个 part 的重建不只看到自己的局部 mask，也能看到它在整体物体中的上下文。

**解决的问题：**  
关节物体中，part 的几何和姿态经常依赖整体结构。例如 drawer_box 的合理形状、位置和滑动方向依赖 board/cabinet；lid 的合理朝向依赖 bottle body。当前独立 part 重建容易出现：局部 mesh 看起来对，但放回整体后相对位置、朝向或连接关系不合理。

**为什么能解决：**  
完整物体上下文提供了邻接 part、主体结构、连接边界和整体尺度信息。把 object-level RGB-D feature 注入 part reconstruction 后，模型/优化器可以区分“这是 part 自身的可见边界”还是“这是与 parent 接触的结构边界”，从而减少 part-cut 泄漏、薄板补全错误和对称 part 的方向歧义。

**要修改哪些代码：**
1. `reconstruction/tools/recon_tsdf_common.py`
   - 在 `_load_frame_observation` 中除了加载当前 part 的 `mask/depth/pose/points_cam`，增加可选的 object-level observation：
     - object mask
     - 同帧其他 part masks
     - part bbox 在 object bbox 中的位置
     - part-to-object relative center/extent
   - 在 `_iterative_tsdf_part` 的 observation 记录中保存这些 context 字段，供后续 frame selection、pose refinement 和 TSDF 权重使用。
2. `reconstruction/tools/recon_part_postprocess_common.py`
   - part cutting 时加入 object-level projection score，避免只根据单个 part mask 切 mesh，导致相邻主体结构误切入或误删除。
3. 新增 `reconstruction/tools/part_context.py`
   - 统一计算：
     - object bbox
     - part bbox
     - part center in object frame
     - part 与其他 part 的 2D/3D 邻近关系
     - object-conditioned confidence

**借鉴来源是什么：**  
CAPNet 的 dataset 和 network 都不是单独裁剪每个 part，而是从完整物体 ROI 中采样点云和语义/实例信息。相关代码：
- `related_works/CAPNet/datasets/datasets_rgbdart.py`：从完整 object mask 得到 ROI，crop RGB、depth、semantic、instance，再采样所有有效 part 点。
- `related_works/CAPNet/networks/capnet.py`：将图像特征按每个 3D 点的像素位置 gather 到点特征上。

**人家的代码如何实现：**  
CAPNet 在 `RGBDArtDataSet.__getitem__` 中读取 `semantic_segmentation` 和 `instance_segmentation`，用 object mask 计算整体 bbox，然后对 RGB、depth、semantic、instance 进行同一个 ROI crop。随后根据 depth 和像素坐标反投影得到点云，并保留每个点对应的 `roi_sem`、`roi_inst` 和像素坐标 `roi_xs/roi_ys`。在 `MainNet.extract_pts_feature` 中，CAPNet 先用 FeatUp/DINOv2/SAM2 提取 ROI 图像特征，再用 `roi_xs/roi_ys` gather 出每个 3D 点对应的 2D feature，最后将 `pts + color + rgb_feat` 输入 PointNet++。这里借鉴的是“完整物体上下文 + part 语义/实例关系 + 2D/3D feature lifting”的数据组织方式，而不是它的标准坐标监督。

---

### 2. Geometry relation graph：不用 GNN，只用几何关系约束局部和整体

**要实现的功能：**  
为每个 object 构建一个显式的 part relation graph。节点是 part，边是 parent-child 或几何邻接关系。第一版不训练 GNN，只用几何约束：

```text
node: part_id, semantic, mesh, local pose, object-relative pose
edge: parent, child, joint type, axis, origin, limit, contact prior
```

该图同时服务 reconstruction 和 downstream ref_pose：
1. reconstruction 阶段：约束 part mesh 的装配关系和相对 pose；
2. pose est 阶段：约束 ref_pose 的独立 part pose，消除对称 part 的若干度旋转歧义。

**解决的问题：**  
关节物体不同于刚体，part 的正确性不能只看局部 mesh。一个 part 单独看可能 pose 正确，但放回整体时会因为自身对称性绕轴旋转若干度，导致与 parent 的连接位置、joint axis 或滑动方向不一致。

**为什么能解决：**  
几何关系图把局部 part pose 投影到合法的 articulated configuration 上。对于 slider，child 只能沿 parent frame 中的 joint axis 平移；对于 hinge，child 只能绕 joint axis 旋转。这样可以排除那些局部渲染误差很小、但整体关系不合法的对称解。

**要修改哪些代码：**
1. 新增 `reconstruction/tools/relation_graph.py`
   - 从 `dataset_train/<split>/<object>/meta.json` 读取 `mobility_path`。
   - 读取 `mobility_v2.json`，解析：
     - `id`
     - `parent`
     - `joint`
     - `jointData.axis.origin`
     - `jointData.axis.direction`
     - `jointData.limit`
   - 结合 `meta.json` 的 `part_layout` 和 `masks/<part_name>`，输出 `relation_graph.json`。
2. 新增 `reconstruction/tools/graph_pose_refine.py`
   - 输入每个 part 的初始 pose。
   - 优化 root pose 和每条 joint 的 q。
   - 输出满足 graph constraints 的 part poses。
3. 修改 `reconstruction/tools/recon_tsdf_common.py`
   - 在 `run_tsdf_object` 开始时加载 relation graph。
   - 在 `_iterative_tsdf_part` 结束后保存 part 的局部 pose、object-relative pose 和 graph node metadata。
   - 对称 part 的 seed/refine 结果可增加 graph consistency score。
4. 与 `ref_pose` 衔接：
   - `ref_pose` 独立输出 `ob_in_cam2/<frame>/pose_XXXX.txt` 后，调用 graph refinement，输出 `ob_in_cam2_graph/<frame>/pose_XXXX.txt`。
   - 后续可把 graph refined pose 作为 `init_pose_overrides` 再跑一轮 FoundationPose。

**借鉴来源是什么：**  
CAPNet 本身没有显式 kinematic graph，也没有 parent-child joint optimization。但它证明了 part pose 不应该只由局部 part mesh 决定，而应该在完整 object context 和 part semantic 下估计。Relation graph 是对 CAPNet 隐式 object context 的显式几何扩展。

**人家的代码如何实现：**  
CAPNet 的 `evaluate` 阶段会在完整 object 点云中同时预测所有 part 的语义和实例，再逐 part 求 pose。也就是说，它没有把每个 part 完全独立拆开，而是在同一个 object observation 中联合理解 part。我们这里不引入 GNN，也不建立标准归一化坐标空间，而是把这种“part pose belongs to an object-level structure”的思想转化成可解释的几何图优化：CAPNet 用 learned feature 隐式提供上下文，我们用 `mobility_v2.json` 和 RGB-D 几何显式提供 parent-child/joint constraints。



# 改进后的最终pipeline叙事
## 现在的代码如何进行叙事

当前 reconstruction 模块的主线可以讲成一个 part-level reconstruction pipeline：给定 reference 帧的 RGB-D、part mask、相机内参和已有的 part pose，先用 SAM3D / Hunyuan3D / InstantMesh 为每个 part 生成一个可渲染的 mesh 先验；然后用多帧 RGB-D 观测通过 TSDF 对这个 mesh 先验进行几何补全和尺度/位姿校正；最后再用 DLMesh 对表面进行细化，输出给下游 `ref_pose` 使用的 `pose_ready_models`。

这条线已经能解决“单帧 reference 看不全 part”的问题：单帧生成模型提供一个完整形状假设，多帧 TSDF 和 DLMesh 再利用真实 RGB-D 观测修正这个假设。当前代码中对应的叙事是：

1. `reconstruction/run/recon_*.py` 负责调用不同 reconstruction family，生成每个 part 的初始 `model.obj`。
2. `reconstruction/tools/recon_tsdf_common.py` 负责读取每个 part 的多帧 mask/depth/pose，做 seed pose 选择、ICP refine、depth consistency gate、TSDF fusion 和 pose utility check。
3. `reconstruction/tools/recon_dmesh_common.py` 负责把 TSDF 后的 mesh 继续交给 DLMesh 做可微表面细化。
4. 最终输出 `pose_ready_models/view_0/model_XXXX/model.obj`，以及 `raw_pose.txt`、reference points 和若干诊断日志，作为下游 pose estimation 的输入。

但是这条线仍然把每个 part 当成近似独立对象处理。它默认只要单个 part 的 mesh 和 pose 在当前 mask/depth 下合理，就可以作为最终 part prior。对于刚体物体这通常够用；但对关节物体不够，因为 part 的正确性不只取决于自己的局部几何，还取决于它和 parent、joint、整体结构之间的关系。

因此当前代码的主要缺口是：

1. part 重建缺少完整 object context。单个 drawer、lid、handle 的局部几何可能看起来合理，但它和主体连接边界、整体尺度、邻接 part 的关系没有被显式利用。
2. part pose 缺少 relation constraint。单独看一个对称 part，绕自身轴旋转若干度可能仍然能和局部 mask/depth 对上；但放回整个 articulated object 后，它可能已经不满足 parent-child 连接、joint axis 或运动方向。
3. 输出给 `ref_pose` 的 init pose 仍然主要来自单 part 几何对齐。下游 FoundationPose/refiner 会受这个 init pose 影响，如果 init 已经落入对称错误解，后续局部 refine 很难自己跳出来。


## 增加了哪些部分

改进后只增加两个核心部分，不引入 GNN，也不把目标改成标准归一化坐标空间。

### 1. Object-conditioned part context

这一部分把 CAPNet 中“在完整 object observation 中理解 part”的思路迁移到 reconstruction。具体做法不是训练 CAPNet，也不是预测 NOCS，而是在现有 TSDF / part-cut 数据流里增加 object-level context：

1. 对每个 frame，除了当前 part mask，还读取 object mask 和其他 part masks。
2. 计算 part bbox 在 object bbox 中的位置、part center 相对 object center 的偏移、part 和其他 part 的 2D/3D 邻近关系。
3. 把这些 context 保存进 observation record，供 frame selection、part-cut projection score、TSDF consistency check 和日志诊断使用。

它对最终 mesh 的影响是：part mesh 不再只由自己的二值 mask 决定，而是会参考它在整体物体中的位置和邻接关系。例如 drawer_box 的可接受边界应靠近 board/cabinet，lid 的合理朝向应和 bottle body 的连接区域一致。这样可以减少相邻主体结构被误切进 part mesh，或者薄结构因为局部 mask 不稳定而被错误补全。

它对 init pose 的影响是：seed/refine 阶段可以加入 object-context score。某个 seed pose 即使局部 depth residual 不差，如果它把 part 放到了不符合整体位置关系的地方，也会被降低优先级。这样下游 `ref_pose` 拿到的 `raw_pose.txt` 或 graph-refined pose 更稳定。

对应新增/修改代码：

1. 新增 `reconstruction/tools/part_context.py`，统一计算 object mask、part/object bbox、相对位置和邻近关系。
2. 修改 `reconstruction/tools/recon_tsdf_common.py`，在 `_load_frame_observation` 和 `_iterative_tsdf_part` 中保存 context，并在 seed/frame selection 的诊断中加入 context score。
3. 修改 `reconstruction/tools/recon_part_postprocess_common.py`，part-cut 时加入 object-level projection score，避免只看单个 part mask。

### 2. Geometry relation graph

这一部分显式建模 part 与整体的结构关系。它不是 learned graph，也不是 GNN，而是一个几何约束层。图可以从 `meta.json` 指向的 `mobility_v2.json` 读取，也可以在没有 mobility 时退化为 RGB-D 几何邻接图。

图中每个节点表示一个 part：

```text
node = part_id, semantic, mesh_path, local_pose, object_relative_pose
```

每条边表示 parent-child 或邻接约束：

```text
edge = parent, child, joint_type, axis, origin, limit, contact_prior
```

#### Relation graph 处理流程

Relation graph 这一段的输入不是单个 part 的 mesh，而是整个 object 的结构信息和已经整理好的 part 列表。它的目标是把“若干孤立 part”组织成“一个有父子关系和关节约束的 articulated object prior”。最终输出是一个对象级的 `relation_graph.json`，供 reconstruction 记录装配关系，也供下游 `ref_pose` 做结构一致性约束。

模块输入包括：

```text
dataset object root
  -> meta.json
  -> mobility_v2.json / mobility.json
  -> masks/<part_name>/
  -> cam_params/<part_name>/<frame>.txt
  -> reconstructed part model names / part ids
```

其中 `meta.json` 用来找到 mobility 文件和 part layout；`mobility_v2.json` 提供 parent-child、joint type、axis、limit 等结构定义；`masks/<part_name>/` 用来确定 reconstruction 中实际有哪些 part；`cam_params` 用来在 reference 帧中估计 parent 和 child 的 rest relative pose。

整体处理流程可以概括为：

```text
object metadata + part folders
  -> find mobility file
  -> parse part layout and mobility items
  -> build graph nodes for reconstructed parts
  -> build graph edges from parent-child joint records
  -> attach joint axis / joint limit / rest relative pose
  -> export relation_graph.json
```

第一步是定位结构来源。系统先读取 object 下的 `meta.json`，优先使用里面的 `mobility_path`；如果没有，就在 object 目录下查找 `mobility_v2.json` 或 `mobility.json`。如果 mobility 文件不存在，graph 仍然可以退化成只包含 part nodes 的 `parts_only` 图。

第二步是建立节点。系统遍历 `masks` 目录中实际出现的 part，把每个 part 转成一个 graph node。每个 node 记录 part id、part 文件夹名、语义名称、mobility 中的名字、parent id 和 joint type。这样 graph 的节点和 reconstruction 实际输出的 `model_XXXX/model.obj` 可以对应起来。

第三步是建立边。系统根据 mobility 文件中的 parent-child 记录，为每个有 parent 的 child part 建立一条 relation edge。edge 记录这个 child 连接到哪个 parent、连接类型是 fixed / hinge / slider 还是其他 joint，以及 parent 是否也在当前 reconstruction 的 part 列表中。

第四步是补充关节约束。对于每条 edge，系统读取 joint axis 的 origin 和 direction，读取 joint limit，并保存到 graph 中。这一步让 relation graph 不只是“谁连着谁”，而是进一步表达“沿哪里转、沿哪里滑、运动范围是多少”。

第五步是补充 reference 下的相对位姿锚点。如果 parent 和 child 在同一个 reference frame 中都有 `cam_params` pose，系统会计算它们在该 reference frame 下的相对位姿，保存为 `rest_relative_pose`。这个字段不是为了固定 inference 帧的绝对朝向，而是提供一个结构锚点：child 在合理装配状态下应该如何相对 parent 放置。

第六步是写出图文件。最终 graph 会保存为对象级 JSON，里面包含：

```text
object / split / source
root_candidates
nodes
edges
```

它的输出可以理解为：

```text
isolated reconstructed parts
  -> structured part graph
  -> articulated object prior
```

简短描述可以写成：

```text
Relation graph reads the object metadata and mobility file, matches them with the actually reconstructed part folders, builds one node for each part, adds parent-child edges with joint type, axis and motion limits, and optionally records the reference-frame relative pose between connected parts. The output is a compact `relation_graph.json` that turns separate part meshes into a structured articulated-object prior.
```

它对最终 mesh 的影响是：mesh 本身不一定被强行变形，但 mesh 的装配解释会被结构化保存。也就是说，输出不再只是若干孤立的 `model.obj`，而是：

```text
part meshes + relation_graph.json + per-part graph metadata
```

这让 reconstruction 的输出从“多个 part mesh 文件”升级为“一个 articulated object prior”。当某个 part mesh 的可见局部很对称时，relation graph 可以解释它应该如何连接到 parent，而不是让它在局部坐标里自由旋转。

它对 init pose 的影响更直接：relation graph 可以对独立 part pose 做 graph refinement。优化变量不是每个 part 的自由 SE(3)，而是 root pose 和每条 joint 的状态 `q`。优化后得到的 part poses 必须满足 slider/hinge/limit/contact 约束：

```text
independent part poses
        ↓
graph pose refinement
        ↓
articulated-consistent part poses
```

这正好解决“单独看 part pose 没问题，放回整体后因为对称性旋转若干度”的问题。局部上合理但整体上违反 joint axis、parent-child 连接或 limit 的 pose 会被 graph energy 排除。

对应新增/修改代码：

1. 新增 `reconstruction/tools/relation_graph.py`，从 `meta.json` / `mobility_v2.json` 构建 `relation_graph.json`。
2. 新增 `reconstruction/tools/graph_pose_refine.py`，输入每个 part 的初始 pose，输出满足几何图约束的 part poses。
3. 修改 `reconstruction/tools/recon_tsdf_common.py`，在 `run_tsdf_object` 开始时加载 graph，并在每个 part 重建结束后保存 graph node metadata。
4. 与 `ref_pose` 约定输出接口：`ref_pose` 可以读取 `relation_graph.json` 和 graph-refined init poses，在独立 FoundationPose/refiner 之后做一次 graph refinement，或者把 graph refined pose 作为 `init_pose_overrides` 再跑一轮。


## 最终的融合叙事

最终 pipeline 可以讲成三段：

```text
Part-aware RGB-D Feature Lifting
              ↓
Context-conditioned Part Reconstruction
              ↓
Geometry Relation Graph-guided Pose and Assembly Optimization
```

第一段，`Part-aware RGB-D Feature Lifting`，负责把 RGB-D observation 从单个 part mask 扩展到完整 object context。每个 part 仍然有自己的 mask、depth 点云和 pose，但这些局部观测会附带 object bbox、邻接 part、relative center/extent 和 object mask 信息。这里借鉴 CAPNet 的不是 canonical coordinate，而是它“在完整 object ROI 中联合组织 part observation”的思想。

第二段，`Context-conditioned Part Reconstruction`，负责生成更可靠的 part mesh。SAM3D / Hunyuan3D / InstantMesh 提供初始 mesh 先验，TSDF 和 DLMesh 继续根据真实 RGB-D 修正几何。不同的是，重建过程不再只问“这个 part 在自己的 mask 里是否拟合”，还会问“这个 part 是否处在整体物体中合理的位置、边界和邻接关系里”。因此最终 mesh 的提升不是来自一个新的生成网络，而是来自更可靠的 frame selection、part-cut 评分、TSDF 接收策略和 object-context 诊断。

第三段，`Geometry Relation Graph-guided Pose and Assembly Optimization`，负责把若干 part mesh 和 part poses 组织成 articulated object prior。它用 `relation_graph.json` 显式保存 parent-child、joint type、axis、origin、limit 和 contact prior。对于 reconstruction，它提供装配一致性诊断；对于下游 `ref_pose`，它提供 pose refinement 约束：独立估计出来的 part pose 必须被投影到合法的关节构型上。

最终输出给下游的不再只是：

```text
model_0000/model.obj
model_0001/model.obj
...
```

而是：

```text
pose_ready_models/view_0/model_XXXX/model.obj
pose_ready_models/view_0/model_XXXX/raw_pose.txt
pose_ready_models/view_0/model_XXXX/graph_node.json
relation_graph.json
graph_refined_init_poses/<frame>/pose_XXXX.txt   # 可选
reconstruction_context.json
```

这样，`ref_pose` 的输入从“孤立 part mesh + 单个 init pose”升级为“part mesh + object context + articulated relation prior”。推理 query 帧仍然只需要 RGB-D、mask 和 K；relation graph、mesh、reference evidence 都是 reconstruction/reference 阶段预先准备好的 object prior，不使用 query frame 的 GT pose 或 GT joint state。

这套叙事的关键点是：我们不把关节物体拆成若干独立刚体，也不强行学习一个标准归一化坐标空间。我们保留 part-level reconstruction 的优势，同时把 object context 和几何关系图加回来，让每个 part 的 mesh 和 init pose 都接受整体结构的约束。这样既能提升 reconstruction 对 part 的稳定性，也能为下游 pose estimation 提供更不容易陷入对称歧义的初始化与后处理约束。

最终一句话可以概括为：

> Reconstruction first builds reliable part-level mesh priors from RGB-D observations, then uses object-level context and an explicit geometry relation graph to turn isolated part priors into an articulated object prior, providing downstream pose estimation with both better meshes and structure-consistent initial poses.


## 与下游 ref_pose 的最终衔接叙事

`reconstruction` 输出的目标不是只给下游若干孤立的 part mesh，而是给 `ref_pose` 一个带结构信息、reference evidence 和 init pose 的 articulated object prior。下游 `ref_pose` 的任务是在 query frame 中，利用这个 prior、query RGB-D/mask 和相机内参，估计每个 part 在 query camera frame 下的最终 pose。

整体衔接可以概括为：

```text
reconstructed part mesh + reference evidence + graph/context prior + query RGB-D/mask + init pose
        ↓
local pose hypothesis generation
        ↓
reference-query correspondence correction
        ↓
reliability-aware FoundationPose refinement
        ↓
depth/mask/reference/graph consistency reranking
        ↓
final part pose
```

### 1. 下游模块输入输出

`ref_pose` 接收的输入包括五类信息。

第一类是 reconstruction 输出的 part mesh：

```text
pose_ready_models/view_0/model_XXXX/model.obj
```

这个 mesh 是可渲染、可优化的几何先验。它补全了单视角 reference 中不可见的部分，但由于来自单视角重建和生成式补全，不能被无条件当成可靠 CAD。

第二类是 reconstruction 保存的 reference evidence：

```text
reference_points.npy
reference_points_obj.npy
reference_points_cam.npy
raw_pose.txt
local_to_object.txt
local_to_reference_camera.txt
```

这些文件记录 reference view 中真实看见过的点、坐标变换和原始参考 pose。它们不完整，但比补全 mesh 更可靠，因为它们来自真实 RGB-D observation。

第三类是 query frame 的观测：

```text
query RGB
query depth
query part mask
camera intrinsic K
```

这是最终判断 pose 是否正确的真实观测依据。

第四类是 init pose。它可以来自 reconstruction 的 `raw_pose.txt`，也可以来自 graph refinement 后的 pose，或者来自 match 阶段输出的 `coarse_pose/refined_pose`。

第五类是结构约束信息：

```text
relation_graph.json
graph_node.json
graph_refined_init_poses/<frame>/pose_XXXX.txt
connector vector / edge gate / kinematic consistency   # optional
```

这些信息用于判断独立 part pose 是否符合整体 articulated object 的 parent-child、joint axis、limit 和 contact 约束。

`ref_pose` 的输出包括：

```text
final refined pose for each part
local sampled candidate poses
correspondence-corrected poses
FoundationPose refined candidate poses
pose_score.json
refine_trace.json
validity / depth / mask / overlay visualizations
```

最终输出不只是一个 pose 文件，而是一组可解释结果：每个 candidate 如何生成、如何校正、如何 refinement、为什么被选中或淘汰，都可以通过日志和 score breakdown 追踪。

### 2. Reconstructed Part Prior Loading

这一部分读取 reconstruction 输出的 `model.obj`。在叙事中，它不是可靠 CAD，而是带不确定性的 generative part prior。

它的工作是给 `ref_pose` 提供一个可渲染、可优化、形状相对完整的几何假设。单帧 reference 只能看到 part 的局部，而 FoundationPose-like render-and-refine 需要完整可渲染模型，所以 reconstruction mesh 是必要输入。

这样设计有效，是因为它缓解了单视角信息不足的问题：reference evidence 提供真实但局部的观测，reconstructed mesh 提供完整但不完全可信的形状先验。后续 pipeline 正是围绕“使用 mesh，但不盲信 mesh”展开。

### 3. Reference Evidence Anchoring

这一部分把 reconstruction 阶段保存的真实 reference evidence 显式引入 `ref_pose`。系统会读取 reference-visible points、reference pose 和 local/object/reference-camera transform。

它的核心工作是区分两类信息：

```text
model.obj: 完整，但可能包含 hallucination
reference evidence: 不完整，但来自真实 RGB-D 观测
```

reference evidence 会用于 candidate 校正、pose verification 和 reranking。具体来说，系统会把 reference-visible points 根据 candidate pose 投影到 query view，再检查这些点是否落在 query mask 内、是否与 query depth 一致。

这样设计有效，是因为它直接针对“单视角重建 mesh 不可靠”的问题。即使 mesh 的补全部分有误，reference-visible points 仍然是可靠锚点，可以防止 refiner 被 hallucinated geometry 带偏。

### 4. Local Pose Hypothesis Generation

这一部分从 init pose 出发，在局部邻域内生成多个 pose candidates。如果上游已经给出 Top-K candidates，就直接使用；如果上游只给一个 init pose，系统会围绕该 pose 做 FoundationPose-like local spherical sampling，生成多个局部旋转和平移假设。

它把单个初始化扩展为候选池：

```text
single init pose
        ↓
local sampled pose candidates
```

每个 candidate 只是一个可能解释，不直接作为最终结果。

这样设计有效，是因为单视角 reference 和 query 之间存在视角歧义、遮挡、对称性和局部形状歧义。单个 init pose 一旦落入错误 basin，后续 refinement 很难跳出来。多假设机制可以覆盖局部不确定性，降低单点初始化失败导致整个 pipeline 失败的风险。

### 5. Reference-query Correspondence-guided Alignment

这一部分对每个 local pose candidate 做中层几何校正。系统会把 reference-visible points 按当前 candidate 投影到 query view，然后利用 query mask 和 query depth 检查这些点是否有合理对应。如果存在足够可靠的 correspondence，就对 candidate 做小范围 camera-space correction。

流程是：

```text
local sampled candidate
        ↓
project reference-visible points to query
        ↓
check mask/depth correspondence
        ↓
estimate small pose correction
        ↓
correspondence-corrected candidate
```

它不替代 FoundationPose，而是在 FoundationPose refinement 前先把 candidate 拉到更合理的位置。

这样设计有效，是因为 FoundationPose 的 render refinement 依赖初值和 mesh 几何。如果 candidate 离真实 pose 太远，或者 mesh 中有错误补全部分，直接 refinement 可能被误导。reference-query correspondence 利用真实 reference observation 做中层约束，让 candidate 在进入 render-based refinement 前已经和 query observation 建立几何对应。

### 6. Reliability-aware FoundationPose Refinement

这一部分把 correspondence-corrected candidates 送入 FoundationPose-like render-and-refine 流程。对每个 candidate，系统渲染 reconstructed mesh，与 query RGB-D crop 对齐，并迭代预测 SE(3) delta。

关键点是这里不是普通的全区域 render matching，而是 reliability-aware refinement。refiner 会输出：

```text
geom_validity
pose_utility
validity_mask = geom_validity * pose_utility
```

`geom_validity` 判断 rendered geometry 是否可信，`pose_utility` 判断该区域是否对 pose update 有帮助，`validity_mask` 控制哪些区域真正参与 refinement。

这样设计有效，是因为 reconstruction mesh 可能包含错误补全区域。如果所有 rendered pixels 都被同等信任，hallucinated region 会误导 pose update。validity-aware refinement 让模型更关注 query 中可见、depth/mask 一致、且对 pose 有判别力的区域，从而降低不可靠几何的影响。

### 7. Observation, Reference and Graph Consistency Reranking

这一部分对所有 refined candidates 统一打分并重排序。最终 pose 不由某一个阶段直接决定，而是由 query observation、reference evidence、refinement trace 和 graph consistency 共同验证。

评分项包括：

```text
rendered mask 与 query mask 的 overlap
rendered depth 与 query depth 的一致性
outside-mask leakage
rendered visible ratio
validity confidence
pose update stability
reference-visible points 的 mask/depth consistency
graph / connector / kinematic consistency
```

最终选择综合 score 最高的 refined candidate 作为 part pose，并保存 score breakdown。

这样设计有效，是因为 pipeline 中每个阶段都可能出错：local sampling 可能产生错误候选，correspondence correction 可能受噪声影响，FoundationPose refinement 可能被错误 mesh 区域带偏。最终 reranking 用真实 query RGB-D/mask、reference evidence 和 relation graph 做统一验证，可以过滤掉与观测或结构不一致的候选。

### 8. 为什么整体有效

这套衔接叙事的关键是让不同信息承担不同角色：

1. reconstruction mesh 提供完整形状先验，但不被当成可靠 CAD。
2. reference evidence 提供真实观测锚点，用来约束生成式 mesh。
3. object context 和 relation graph 提供整体结构约束，避免独立 part pose 落入对称歧义。
4. local hypothesis generation 处理单个 init pose 的不确定性。
5. correspondence-guided alignment 在 FoundationPose 前做中层几何校正。
6. reliability-aware refinement 在像素/区域层面降低错误几何的影响。
7. consistency reranking 在 pose 层面选择最符合真实观测和结构关系的结果。

对于“单视角信息不足”的问题，pipeline 不要求单个 reference 或单个 init pose 一步到位，而是通过完整 mesh prior、多 pose hypotheses、reference-query correspondence 和 query observation 逐步收敛到合理 pose。

对于“单视角重建 mesh 不可靠”的问题，pipeline 不直接相信 mesh，而是用 reference evidence 标定真实可见几何，用 validity mask 降低不可靠区域权重，用 depth/mask/reference/graph consistency 最终验证每个 candidate。

最终一句话可以概括为：

> `ref_pose` treats the reconstructed part mesh as an uncertain generative prior, uses reference evidence and object-level relation constraints to correct and verify local pose hypotheses, and finally selects the pose that best agrees with query RGB-D, mask, reference observations and articulated structure.


# 修改日志



# dlmesh部分处理流程

DLMesh 这一段接在 base reconstruction 或 TSDF 后面，目标不是重新生成一个新模型，而是在已有 mesh 的基础上，用多帧 RGB-D 观测把表面进一步拉到更贴近真实深度和 mask 的位置。它可以理解成一个可微的 mesh polishing 阶段：输入是一个已经大致对齐的 part mesh，输出是几何更干净、表面更贴近观测、仍然保持原始拓扑结构的 `model.obj`。

## 1. 从输入到输出，mesh 经历了什么

输入包括：

```text
base mesh / TSDF mesh
part masks
RGB-D frames
camera intrinsics
current part pose observations
```

处理流程可以概括为：

```text
base mesh
  -> mesh cleanup / simplification
  -> load valid RGB-D keyframes
  -> estimate mesh-observation alignment for each keyframe
  -> select reliable and diverse keyframes
  -> stage-wise differentiable vertex optimization
  -> export refined model.obj + summary log
```

第一步是 mesh 预处理。系统先读取 base mesh，清理退化面、重复面和未被引用的顶点；如果 mesh 有很多碎片，会保留主要连通部分；如果面数太多，会做一次简化。这样做的目的不是改变物体形状语义，而是让后续可微优化在一个更稳定、更轻的 mesh 上进行。

第二步是读取观测帧。对于每个 part，系统遍历可用 RGB-D frame，读取 part mask、depth 和相机内参，并从 mask 内的 depth 反投影出当前帧的局部点云。mask 太小、有效深度点太少的帧会被跳过。

第三步是为每个候选帧估计 mesh 和观测之间的对齐关系。系统先用 ICP 把当前帧的 depth 点云对齐到 mesh，再用一个轻量的 pose optimization 继续细化这个对齐关系。对齐误差过大的帧会被丢弃。

第四步是选择 keyframes。系统不会把所有帧都塞进优化，而是保留一组可靠且视角有差异的帧。太接近的帧信息重复，太远的帧容易带来错误对齐；因此 keyframe pool 会同时考虑对齐质量、视角间隔和平移间隔。

第五步是分阶段优化 mesh 顶点。系统把选中的 keyframes 分批加入，逐步更新 mesh 顶点位置。优化时会同时考虑四类信号：

1. mesh 表面要靠近 mask/depth 反投影得到的真实观测点。
2. mesh 投影到图像后，深度要和观测 depth 一致。
3. mesh 投影不要大量跑到 part mask 外面。
4. 顶点不要偏离原始 mesh 太远，局部表面也要保持平滑。

第六步是导出结果。优化后的顶点和原 mesh 的面连接关系重新组成 refined mesh，清理后写成最终的 `model.obj`，同时保存每一轮 keyframe 选择、对齐质量和优化 loss 的 summary 日志。

## 2. 简短 pipeline 叙事

可以这样描述 DLMesh 阶段：

```text
DLMesh takes the coarse part mesh from reconstruction or TSDF, cleans and simplifies it, selects reliable RGB-D keyframes, aligns each keyframe to the mesh, and then optimizes the mesh vertices so that the surface agrees with observed depth and part masks while staying close to the original shape prior. The output is a refined pose-ready mesh for downstream pose estimation.
```

中文版本可以写成：

```text
DLMesh 阶段以 reconstruction/TSDF 得到的粗 mesh 为输入，先做网格清理和简化，再从多帧 RGB-D 中选择可靠且视角互补的 keyframes；每个 keyframe 先与 mesh 做位姿对齐，然后共同约束 mesh 顶点的可微优化，使表面更贴近真实 depth 和 part mask，同时保持原始 mesh 的整体形状和平滑性。最终输出 refined `model.obj`，作为下游位姿估计的 pose-ready mesh。
```

更短的流水线版本：

```text
coarse mesh
  -> cleanup and simplification
  -> RGB-D keyframe selection
  -> keyframe-to-mesh pose alignment
  -> differentiable vertex refinement
  -> refined pose-ready model.obj
```

# Relation Graph








# todo

## 1. 在 sam3d_partcut_tsdf_dmesh 后接入 part relation graph

目标：让 `sam3d_partcut_tsdf_dmesh` 的最终输出不只是若干独立的 part mesh，而是一个带有 object-level 结构关系的 articulated object prior。输出中需要显式包含 part 与 object、part 与 parent/child 之间的相对关系，供后续 `ref_pose` 使用。

当前 pipeline 是：

```text
SAM3D base
  -> reference-only PartCut
  -> TSDF refinement
  -> DLMesh refinement
  -> pose_ready_models/view_0/model_XXXX/model.obj
```

计划改成：

```text
SAM3D base
  -> reference-only PartCut
  -> axis alignment
  -> TSDF refinement
  -> DLMesh refinement
  -> coordinate-frame consolidation
  -> relation graph construction
  -> structured reconstruction output
```

Relation graph 接在 DLMesh 后面做对象级导出。原因是 DLMesh 之后的 `model.obj` 才是最终要交给下游的 pose-ready mesh，graph 里记录的 node、edge、relative pose 和 joint prior 应该对应最终 mesh 坐标系，而不是中间 mesh 坐标系。

输入：

```text
dataset object root
  -> meta.json
  -> mobility_v2.json / mobility.json
  -> masks/<part_name>/
  -> cam_params/<part_name>/<frame>.txt
  -> final pose_ready part meshes
  -> optional axis_alignment.json
```

输出：

```text
<work_root>/sam3d_partcut_tsdf_dmesh/<split>/<object>/
  relation_graph.json
  graph_node.json
  pose_ready_models/view_0/model_XXXX/graph_node.json
```

处理流程：

```text
final part meshes + metadata
  -> read meta / mobility
  -> match reconstructed model_XXXX with part ids
  -> build graph nodes
  -> build parent-child / joint edges
  -> attach joint type, axis, limit and rest relative pose
  -> attach final mesh coordinate-frame metadata
  -> export relation_graph.json and per-part graph_node.json
```

实现上优先复用已有的 `reconstruction/tools/relation_graph.py`，不要重新写一套 graph 解析逻辑。需要补的是 pipeline 接入和输出整理：在 `recon_sam3d_partcut_tsdf_dmesh.py` 的 stage 3 之后增加 stage 4，调用 `write_relation_graph()`，并把每个 part 对应的 graph node 写到该 part 的 pose-ready 目录里。

需要注意：relation graph 不应该改变 mesh 本身，也不应该影响 TSDF/DLMesh 的优化过程。第一版只做结构描述和下游接口输出，避免把 graph optimization 提前塞进 reconstruction 主流程导致结果不可控。

## 2. 增加类似 Any6D 的 axis alignment

目标：在 TSDF 和 DLMesh 之前，把生成式 mesh 的坐标轴、尺度和 reference RGB-D observation 做一次粗对齐，减少由于生成式模型坐标系不稳定带来的轴置换、尺度偏差和 mirror-like 错误。

建议接入位置：

```text
PartCut output
  -> axis alignment
  -> TSDF
  -> DLMesh
```

原因：TSDF 和 DLMesh 都需要初始 mesh 与真实 RGB-D observation 大致对齐。如果初始 mesh 的主轴方向或尺度已经错了，后面的融合和微分优化会更容易被带偏。

每个 part 的输入：

```text
part-cut mesh
reference RGB-D
part mask
camera intrinsics K
reference part pose
object mask / object bbox
optional relation graph prior
```

处理流程：

```text
part mask + depth
  -> reference-frame part point cloud
  -> depth erosion / bilateral filtering / outlier removal
  -> observed OBB

part-cut mesh
  -> surface point sampling
  -> mesh OBB

observed OBB + mesh OBB
  -> enumerate axis permutations and axis signs
  -> reject mirrored transforms
  -> generate axis-aligned mesh candidates
  -> score by mask projection, depth consistency and object-context consistency
  -> select best alignment
  -> export aligned mesh and alignment metadata
```

输出：

```text
pose_ready_models/view_0/model_XXXX/
  model.obj
  axis_alignment.json
```

`axis_alignment.json` 记录这次对齐做了什么，而不是只覆盖 mesh：

```json
{
  "source": "sam3d_partcut",
  "target": "reference_observation",
  "axis_permutation": ["x", "z", "y"],
  "axis_sign": [1, -1, 1],
  "scale": [1.02, 0.98, 1.01],
  "mesh_to_aligned": [
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0
  ],
  "score": {
    "mask_iou": 0.84,
    "depth_consistency": 0.76,
    "object_context": 0.91
  }
}
```

这里要特别避免直接使用反射变换。axis sign 可以枚举，但最终旋转必须保持 right-handed coordinate frame。否则 mesh 局部看起来可能对齐了，实际会把 part 坐标系镜像掉，后续 pose estimation 仍然容易出现正反跳变。

## 3. 坐标系和 graph 的衔接原则

Axis alignment 会改变 part mesh 的坐标系，因此 relation graph 必须在 axis alignment 之后构建，或者至少在 graph 中记录每个 part 的 `mesh_to_aligned`。推荐第一版采用更简单稳定的设计：

```text
axis alignment writes final pose-ready mesh
relation graph reads final pose-ready mesh metadata
relation_graph.json records final mesh coordinate frame
```

这样下游 `ref_pose` 只需要读取最终的 `model.obj`、`axis_alignment.json` 和 `relation_graph.json`，不需要理解中间的 PartCut / TSDF / DLMesh 坐标系。

## 4. 最小实现顺序

1. 先新增 axis alignment 工具，只处理单个 part mesh 和 reference observation，输出 aligned mesh 与 `axis_alignment.json`。
2. 在 `sam3d_partcut_tsdf_dmesh` 中把 axis alignment 放在 PartCut 和 TSDF 之间。
3. 复用现有 `relation_graph.py`，在 DLMesh 后增加对象级 graph 导出。
4. 给每个 `model_XXXX` 写一份 `graph_node.json`，方便下游按 part 读取。
5. 最后再考虑 graph-guided pose refinement，不放进第一版 reconstruction 主流程。

# Any6D可借鉴内容

## 1. OBB-based axis alignment

Any6D 在 `register_any6d()` 中会从 query depth 和 mask 得到 object point cloud，计算观测点云的 OBB；同时从 mesh 采样点云并计算 mesh OBB。随后用 OBB 主轴方向做粗旋转对齐，并用 extent 匹配估计尺度。

可以借鉴到我们的 reconstruction 中：对每个 part，在 TSDF 前用 reference RGB-D point cloud 和 part mesh 做 OBB 对齐，先消除明显的轴置换和尺度错位，再进入 TSDF / DLMesh。

## 2. 深度预处理和点云清洗

Any6D 在 pose 前会做：

```text
depth erosion
bilateral depth filtering
statistical outlier removal
normal estimation
```

这些操作适合放到我们的 axis alignment 和 TSDF keyframe selection 前。作用是减少 mask 边缘毛刺、深度空洞和离群点对 OBB、ICP、depth consistency 的影响。

## 3. coarse estimate -> refinement -> scorer 的闭环

Any6D 不是只生成一个初始 pose，而是：

```text
coarse pose candidates
  -> refiner
  -> scorer
  -> select best candidate
  -> optional second-stage refinement
```

这个工程模式可以借鉴到 reconstruction 的 alignment 阶段：axis alignment 不只选一个 OBB 解，而是生成多个轴排列/符号候选，用 mask projection、depth consistency 和 object context 打分，选择最稳定的候选。

## 4. mesh reset / mesh tensor cache

Any6D 在 mesh 被缩放、对齐或更新后，会调用 `reset_object()` 重新生成 mesh tensor、normal、diameter 等缓存。这一点很重要。

我们的 pipeline 如果在 PartCut 后修改 mesh 坐标，也需要同步刷新：

```text
mesh vertices
mesh normals
mesh bbox / diameter
sampled surface points
render cache if any
alignment metadata
```

否则后续 TSDF、DLMesh 或 pose estimation 可能读到旧的 mesh 统计量。

## 5. 多阶段 debug artifact

Any6D 会导出中间 mesh 和可视化，例如 coarse aligned mesh、final mesh、score visualization。这个习惯很适合我们的 reconstruction，因为 axis alignment 一旦错了，后面的 TSDF/DLMesh 都会被带偏。

建议输出：

```text
model_axis_aligned.obj
axis_alignment.json
axis_alignment_debug.png
axis_alignment_candidates.json
projection_before.png
projection_after.png
```

这样可以快速判断错误来自 SAM3D 生成、PartCut、axis alignment、TSDF 还是 DLMesh。

## 6. 可开关的工程参数

Any6D 的运行脚本里有 `--no-axis-align`、`--no-refinement`、`--no-coarse-est` 这类开关。我们也应该给 reconstruction 加类似开关，方便 ablation：

```text
--axis-align / --no-axis-align
--axis-align-debug
--axis-align-score-thresh
--axis-align-use-object-context
```

这样可以单独比较：

```text
sam3d_partcut_tsdf_dmesh
sam3d_partcut_axisalign_tsdf_dmesh
sam3d_partcut_axisalign_graph_tsdf_dmesh
```

## 7. 对称性候选处理

Any6D 在 pose hypothesis 中考虑 symmetry transforms，避免把几何上等价的旋转当成完全错误。我们的 part reconstruction 也会遇到类似问题，尤其是盒盖、抽屉面板、圆柱手柄等局部对称 part。

可以借鉴的不是直接使用 Any6D 的 symmetry table，而是它的候选思想：对存在对称性的 part，不要只保留一个 axis alignment 解，而是保留若干高分候选，再用 object relation graph、joint axis 和 parent-child connection 做最终选择。


# 后续改进方案

## 0. 对当前新增模块的预期

从 `results/recon/res.md` 看，当前最好的是：

```text
sam3d_tsdf_dmesh
  chamfer = 0.114674
  F1@0.01 = 0.057394
```

它相对 `sam3d` / `sam3d_tsdf` 的提升很小：

```text
sam3d          chamfer = 0.114826, F1 = 0.056878
sam3d_tsdf     chamfer = 0.114792, F1 = 0.056876
sam3d_tsdf_dmesh chamfer = 0.114674, F1 = 0.057394
```

这说明当前瓶颈不是“有没有一个后处理模块”，而是 TSDF/DLMesh 对最终几何的有效修正还不够强。新增的 relation graph 和 axis alignment 对量化结果的影响要分开看：

1. Relation graph 主要提升 pose / assembly consistency，不直接改变 mesh 表面，所以对 Chamfer 和 F1@0.01 不会有稳定直接提升。
2. Axis alignment 可能提升 TSDF/DLMesh 的输入质量，尤其是轴置换、尺度不稳定、对称 part 坐标系不稳的 case；但它也可能在 OBB 歧义时选错轴，导致局部几何变差。
3. 因此新增模块更适合作为“让后续融合更稳定”的基础，不应该预期只加 graph/axis alignment 就显著刷高 Chamfer/F1。

想把这两个几何指标明显刷上去，后续应该重点改 **观测选择、TSDF 融合质量、DLMesh 几何损失、候选回退机制**。

## 1. 第一优先级：做 per-part best-of-N 几何候选选择

目前不同方法在表中表现接近，说明某些 part 上 SAM3D 原始 mesh 已经很好，TSDF/DLMesh 反而可能只在一部分 part 上有效。如果统一把所有 part 都送进 TSDF/DLMesh，很容易出现“少数提升、少数变差，整体平均只涨一点”的结果。

建议把每个 part 的输出改成 best-of-N：

```text
candidate 0: sam3d
candidate 1: sam3d_partcut
candidate 2: sam3d_partcut_axisalign
candidate 3: sam3d_tsdf
candidate 4: sam3d_partcut_axisalign_tsdf
candidate 5: sam3d_partcut_axisalign_tsdf_dmesh

reference RGB-D / multi-frame RGB-D
  -> mask projection score
  -> depth consistency score
  -> observed point Chamfer proxy
  -> silhouette leakage penalty
  -> select best candidate per part
```

输出仍然写到一个最终 method，例如：

```text
sam3d_bestof_tsdf_dmesh
```

这样能避免 TSDF/DLMesh 变差的 part 被强行采用。这个策略最可能直接提升 Chamfer 和 F1，因为它不要求每个后处理阶段都单调变好，而是逐 part 选择更好的几何。

## 2. 第二优先级：把 TSDF 从“能融合”改成“只融合高收益帧”

当前 TSDF 已经有 pose utility gate，但结果显示 `hunyuan3d_tsdf`、`instantmesh_tsdf` 甚至会变差，说明帧选择和融合验收仍然偏宽。

参考 Any6D / BundleSDF 的思路，TSDF 前应该更严格地区分好帧和坏帧：

```text
candidate frame
  -> depth valid ratio
  -> mask area / object area sanity check
  -> ICP residual after pose refinement
  -> projected mesh-mask IoU
  -> depth inlier ratio
  -> view novelty
  -> accept only if score improves current mesh
```

具体改法：

1. 提高 `pose-utility-min-score-delta`，让融合必须带来正收益，而不是允许小幅下降。
2. 降低 `pose-utility-max-frame-drop`，避免某一帧融合后把已有好视角破坏。
3. 对 seed frame 做保护：如果 TSDF 输出相对 seed/reference 的 projection score 下降，则回退到 base mesh。
4. 记录每个 part 的 `tsdf_acceptance_report.json`，用于定位哪些帧导致指标下降。

目标不是融合更多帧，而是只融合能提高表面质量的帧。

## 3. 第三优先级：DLMesh 增加“只允许小幅修正”的保守模式

当前 DLMesh 默认会优化顶点位置，但从结果看，Hunyuan/InstantMesh 的 dmesh 可能大幅变差。这说明 DLMesh 有时在噪声观测或错误 pose 下过拟合。

建议新增 conservative DLMesh preset：

```text
--dlmesh-prior-weight higher
--dlmesh-laplace-weight higher
--dlmesh-depth-weight lower at first
--dlmesh-max-vertex-step smaller
--dlmesh-point-trim-quantile lower / more robust
```

处理流程：

```text
base mesh
  -> conservative DLMesh refinement
  -> evaluate against observation proxy
  -> if proxy worse, rollback to input mesh
```

也可以借鉴 nvdiffrec 的 two-pass 思路：

```text
pass 1: optimize pose / scale / coarse alignment only
pass 2: freeze pose, optimize vertices with strong regularization
```

这样 DLMesh 不会同时解决 pose 错、尺度错、表面错三个问题，优化会稳定很多。

## 4. 第四优先级：补全局部缺失区域，而不是只贴合可见表面

Chamfer 和 F1@0.01 都会惩罚完整形状缺失。TSDF 只依赖可见 RGB-D，很容易把单视角不可见区域削薄或留空；SAM3D 生成式 prior 反而能提供完整性。

因此不要让 TSDF 完全替代 generative mesh，而应该做 prior-guided fusion：

```text
generative mesh prior
  -> mark observed-supported surface
  -> mark unobserved but prior-confident surface
  -> TSDF only updates observed-supported area
  -> keep prior surface in unobserved area
```

可实现为：

1. 对 mesh faces 计算 multi-frame visibility。
2. 只有 visibility 足够高且 depth residual 明确的 face 允许被 TSDF/DLMesh 大幅移动。
3. 不可见区域加更强 prior / laplacian / shape preservation。

这比单纯“多帧融合”更适合 articulated object 的 part reconstruction，因为很多 part 在 reference/query 中就是长期部分可见。

## 5. 第五优先级：引入 mask/refinement 的质量控制

Any6D 里大量使用 depth erosion、bilateral filtering、statistical outlier removal，这些操作对重建指标很关键。我们的 TSDF 和 axis alignment 也应该系统化做 depth/mask 清洗。

建议新增一个统一的 observation cleanup：

```text
raw depth + part mask
  -> erode mask boundary
  -> remove small mask islands
  -> bilateral filter depth
  -> remove depth outliers by local median
  -> backproject clean points
```

并把清洗后的点云同时用于：

```text
axis alignment
TSDF frame selection
ICP pose refinement
DLMesh keyframe construction
```

这类改动通常对 F1@0.01 更敏感，因为 0.01 阈值很小，mask 边缘毛刺和深度飞点会直接降低近表面命中率。

## 6. 第六优先级：针对对称 part 保留多个高分候选

Any6D 的一个可借鉴点是不要过早相信单个 pose / alignment，而是通过 scorer 选择候选。我们的 axis alignment 也应该保留多个 candidate，尤其是对称 part：

```text
axis candidate A
axis candidate B
axis candidate C
  -> run cheap projection/depth score
  -> keep top-k
  -> optional run TSDF/DLMesh on top-2
  -> final best-of-N selection
```

这样可以减少 OBB 主轴歧义造成的错误。一旦 axis alignment 选错，后续 TSDF/DLMesh 很难救回来；保留 top-k 再用观测验证会更稳。

## 7. 推荐实验顺序

建议先不要一次性堆所有模块，而是按下面顺序做 ablation：

```text
E0: sam3d_tsdf_dmesh baseline
E1: sam3d_partcut_axisalign_tsdf_dmesh
E2: E1 + TSDF stricter positive-gain gate
E3: E2 + conservative DLMesh + rollback
E4: E3 + best-of-N per-part selection
E5: E4 + prior-guided visible/unvisible face preservation
```

预期收益排序：

```text
best-of-N per-part selection
  > TSDF stricter gate / rollback
  > conservative DLMesh
  > prior-guided preservation
  > axis alignment
  > relation graph
```

Relation graph 对 Chamfer/F1 的直接收益最低，但对下游 pose estimation 的稳定性最重要；几何指标要刷上去，优先做候选选择和回退机制。

