# 主要功能
在reference帧中，使用SAM进行分割，得到part语义掩码候选。然后使用大语言模型从这些候选mask中选出真正的part mask。然后用这些选出来的mask进行image-to-mesh重建。
在inference帧中，先用SAM做分割，得到part的掩码候选，然后用重建出来的mesh和reference中的ROI区域进行匹配，选出inference帧中的part mask。

最终得到inference帧中要估计的part的mask。


# 主要解决的问题
1. 合理设置大模型的prompt，从单帧reference中准确筛选出目标part的mask。
2. 将reference中选出的part和inference中的part对应起来，提取出inference中的part。只
3. 选出inference帧能看到的part，不要误匹配。
4. 不能出现一对多或多对一的情况。


# Related work

## 1. Segment Anything 系列：把 part 候选生成做稳

**Segment Anything (SAM)** [Kirillov et al., ICCV 2023] 是当前 pipeline 中 reference/inference 候选 mask 的基础。它的优势是 zero-shot、无需类别训练即可产生大量高质量 region proposals，适合 PartNet 这类类别跨度大、part 形态差异大的场景。当前模块已经使用 SAM 自动生成候选，因此后续提升重点不应只放在“是否用 SAM”，而应放在候选生成后的过滤与重排：候选过碎、候选覆盖整物体、多个候选重叠，以及 function part 与普通几何区域混在一起的问题。

**SAM 2** [Ravi et al., 2024] 将 image segmentation 扩展到 video/streaming memory，适合解决 reference 到 inference 的跨帧一致性问题。如果 inference 帧来自同一个物体序列，可以考虑用 SAM2 的 temporal propagation 或 memory attention 替代每帧独立 SAM，再结合现有 mesh/DINO 打分做校验。这样可以减少相邻帧 part mask 抖动和一对多重复匹配。

**Semantic-SAM / SEEM / Grounded-SAM** 这一类 open-vocabulary segmentation 工作把 SAM proposal 与文本/语义提示结合起来。它们对当前模块的启发是：reference 帧中不必只依赖 VLM 对每个候选做二分类，也可以先用 open-vocabulary grounding 将候选限制在 handle, knob, button, switch, key, drawer, lid, door 等可操作 part 类别附近，再交给 VLM 或 DINO rerank。这样能降低 VLM prompt 的压力，并减少静态表面、logo、纹理区域误选。

## 2. VLM/LLM for visual reasoning：改进 reference part 筛选

当前 reference 选择使用 VLM 判断 SAM mask 是否是 precise, complete, user-operable articulated part。相关工作主要来自 vision-language models 的视觉指令跟随和区域级理解。

**GPT-4V / GPT-4o, Gemini, Qwen-VL, LLaVA, Kosmos-2** 等多模态大模型证明了 VLM 能用自然语言规则对局部图像区域进行语义判断。它们与当前问题直接相关，因为 reference mask 的难点不是单纯语义分类，而是同时判断：mask 是否完整、是否只覆盖一个 part、是否属于可操作部件、是否不是整物体/静态表面。当前 `segmentation/check.py` 的 prompt 已经体现了这个方向。

可以重点借鉴 **Set-of-Mark prompting / visual prompting** 的思路：把多个候选 mask 同时编号显示在一张图中，让 VLM 在全局上下文里选择目标 part，而不是对每个 mask 独立二分类。独立二分类容易出现多个候选都被判 True，导致后续一对多；集合式选择更适合加入“每个语义 part 只能选一个最完整 mask”的约束。对于你的目标 1 和 4，这比单 mask 判别更贴近最终需求。

**Part-level VLM grounding** 相关工作，如 Ferret、Kosmos-2、Shikra、LISA 等，关注区域、指代和 segmentation 的结合。它们说明 VLM 不只适合回答 yes/no，也可以输出 region-level grounding 或解释。对当前模块的可用改进是：让 VLM 输出结构化 JSON，包括 `category`, `is_complete`, `is_single_part`, `reject_reason`, `confidence`，再把这些字段作为后续 DINO/mesh 匹配的先验权重，而不是只保留 True/False。

## 3. Object/part matching with foundation features：解决 reference 与 inference 对应

**DINO / DINOv2** [Caron et al., ICCV 2021; Oquab et al., TMLR 2024] 学到的自监督视觉特征具有较强的跨实例、跨视角语义一致性。当前 `dino_match` 已经用 DINOv2 风格特征做 reference/mesh 与 inference candidate 的匹配，这是合理路线。DINOv2 的价值在于它比 CLIP 更适合 dense/local correspondence，尤其是 handle、door、drawer 这类局部结构。

**SAM6D** [Lin et al., 2024] 将 SAM proposal、DINOv2 matching 与 6D object pose estimation 结合起来，是当前 baseline 的强相关工作。它的关键启发是：先用 SAM 生成 object proposals，再使用模板/渲染视角与图像候选做语义和外观特征匹配。你的模块可以把 SAM6D 从“object-level template matching”迁移到“part-level template matching”：每个由 reference mask 重建出的 part mesh 作为一个 CAD/template，inference 中的 SAM candidate 作为 proposal，然后用 DINOv2 score 进行初筛。

**CNOS: Category-level Novel Object Segmentation** [Nguyen et al., 2023/2024] 也很相关。CNOS 使用 SAM 生成 proposal，并用渲染模板和 DINOv2/CLIP 特征实现无需训练的 novel object segmentation。它对当前任务的意义是提供了一个成熟的 proposal-template matching 框架：模板多视角渲染、proposal 特征池化、相似度排序、NMS/one-to-one 后处理。你的 `direct_match_cnos.py` 已经体现了这个方向，可以作为和 SAM6D 并列的 related baseline。

**FoundPose / MegaPose / Gen6D / OnePose / ZeroPose** 等 work 关注少样本、模板或 CAD 辅助的 6D pose/instance matching。虽然它们多是 object-level，但它们的 render-and-compare 思路能帮助当前模块判断“inference 帧是否真的看到了这个 part”。尤其是对目标 3，可以借鉴这些方法中的 visibility reasoning、multi-view template coverage 和 pose consistency，而不是只看单一 DINO 相似度。

## 4. Single-view or sparse-view reconstruction：把 reference mask 变成可验证的 3D part

当前模块将 VLM 选出的 reference part mask 送入 image-to-mesh/SAM3D 重建，再用重建 mesh 辅助 inference 匹配。相关方向包括 **SAM3D / SAM 3D Objects**, **Wonder3D**, **One-2-3-45**, **Zero123/Zero123++**, **TripoSR**, **InstantMesh**, **LGM** 等单图或少图 3D 重建方法。

这些工作的共同价值是：从单帧 mask 恢复可渲染 3D asset，使后续能做多视角 render matching、可见性判断和几何一致性验证。对当前问题最关键的不是 mesh 是否达到高保真，而是 mesh 是否能稳定产生 part 的多视角 silhouette/template。因此可以优先关注 silhouette consistency、view coverage、scale normalization 和 mask-to-mesh robustness，而不是完整 texture quality。

**NeRF/3D Gaussian Splatting from sparse views** 也可以作为替代方向。如果同一个 object 有多帧 reference，使用多视角 mask 约束重建 part geometry，通常会比单帧 image-to-mesh 更稳定。它能缓解单帧 reference 中遮挡、part 厚度不可见、背面形状幻觉等问题，从而提升后续 mesh render score 的可靠性。

## 5. Articulated object and functional part understanding：定义“哪些 part 该被选中”

你的 VLM prompt 将目标限制在 handle, knob, button, switch, key, drawer, lid, door，本质上是在做 functional/articulated part discovery。相关数据集和方法包括 **PartNet-Mobility** [Xiang et al., 2020], **GAPartNet** [Geng et al., 2023], **Where2Act** [Mo et al., 2021], **Where2Explore**, **UMPNET/RPM-Net style articulation perception** 等。

这些 work 的价值是提供了 part taxonomy、可操作性定义和 articulation prior。当前模块容易误选内部隔板、静态面板、装饰纹理或整物体区域；articulated-part literature 中的 affordance 和 mobility prior 可以转化成 VLM prompt 规则或后处理规则。例如 drawer/door/lid 应该通常是较完整的可运动闭合面，handle/knob/button/switch/key 应该是直接交互部件，内部结构、支撑件和连接件应被排除。

**GAPartNet** 尤其值得对比，因为它把不同类别物体的可泛化 articulated parts 归纳成少量 part classes，并关注跨类别泛化。你的模块目标也是在不重新训练类别专用分割器的情况下识别可操作 part，因此 GAPartNet 可以作为 problem setting 和 part definition 的重要 related work。

## 6. Assignment, visibility, and duplicate suppression：解决误匹配与一对多/多对一

当前目标 3 和 4 属于 proposal matching 的决策层问题。相关经典方法包括 **bipartite matching / Hungarian assignment**, **NMS/Soft-NMS**, **set prediction** 思路，以及 DETR 类方法中的 one-to-one matching 训练/推理约束。虽然这些不是单独的视觉 foundation model，但非常适合当前模块：把所有 candidate-part score 组成矩阵，加入可见性阈值、mask overlap penalty、类别/part prior，再用全局最优的一对一 assignment 代替逐 part 贪心选择。

对于“选出 inference 帧能看到的 part，不要误匹配”，可借鉴 6D pose 和 tracking work 中的 visibility-aware scoring：如果某个 reconstructed part 在当前视角的 render silhouette 与任何 SAM candidate 都没有足够重叠，或者 DINO score 高但 mask 几何/面积/位置明显异常，应允许该 part 输出 `not visible`，而不是强制匹配。这个机制与当前 adaptive rerank 中的 render score 很契合，可以进一步扩展为显式的 no-match option。

## 7. 最值得优先对比或吸收的工作

1. **SAM / SAM2**：作为候选 mask 生成和跨帧一致性的基础。
2. **SAM6D**：最接近当前的 SAM proposal + DINOv2/template matching 路线，可作为主要 baseline。
3. **CNOS**：提供 zero-shot proposal-template segmentation 框架，适合 part mesh/render template matching。
4. **DINOv2**：作为跨视角 part correspondence 的核心视觉特征。
5. **GAPartNet / PartNet-Mobility**：定义 articulated/functional part taxonomy，帮助规范 VLM prompt 和评估目标。
6. **SAM3D / Wonder3D / TripoSR / InstantMesh**：为 reference mask 到 part mesh 的重建和多视角 render verification 提供技术依据。
7. **Set-of-Mark / region-level VLM grounding**：用于改进 reference 帧多候选 mask 的全局选择，减少重复 True 和不完整 mask。

综合来看，最贴合当前模块的一条 related-work 叙事是：SAM/SAM2 负责开放集合候选生成，VLM/region grounding 负责从 reference 中筛出功能性 articulated part，单图/少图 3D 重建把 reference mask 转成可渲染 part template，DINOv2/SAM6D/CNOS 风格匹配负责在 inference 帧中找对应 candidate，最后用 visibility-aware render score 与一对一 assignment 处理不可见 part 和重复匹配问题。
