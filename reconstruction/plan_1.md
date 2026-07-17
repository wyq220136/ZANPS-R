## 提高 TSDF 成功率的改进方向

### 2.1 seed top-k 分支，而不是单 seed

TSDF 成败高度依赖 seed。理想观测下，也可能因为 seed frame 的几何可约束性差而失败。

建议：

```text
top-k seed frames
  -> each seed initializes one fusion branch
  -> run coarse pose refinement
  -> build small candidate TSDF
  -> score each branch
  -> choose best seed branch
```

seed 排序不要只看 mask 面积，可以加入：

```text
visible 3D extent
point cloud non-planarity
surface normal diversity
part bbox coverage
viewpoint diversity
projection/depth consistency with base mesh
```

这样可以避免选到“面积大但几何退化”的 seed。

### 2.2 view selection 从 angle gate 改成 information gain

当前 view gate 主要基于 pose angle / translation。更合理的是估计该帧能提供多少新几何信息：

```text
candidate frame
  -> project current mesh / accepted frames
  -> estimate overlap region
  -> estimate newly visible surface region
  -> compute information gain
  -> accept high-gain frames
```

新的 selection score 可以是：

```text
score = overlap_consistency + new_surface_gain + pose_confidence - redundancy_penalty
```

这样近视角但能补关键区域的帧不会被误拒，远视角但无法稳定配准的帧也不会被盲目接收。

### 2.3 consistency 分成 overlap consistency 和 new-surface utility

把双向 depth consistency 拆开：

```text
overlap consistency:
  accepted 已经覆盖的区域必须和 candidate 一致

new-surface utility:
  accepted 未覆盖但 candidate 可见的区域，只要在 candidate 自身中稳定，就作为新增信息
```

处理流程：

```text
candidate frame
  -> find overlap with accepted fused surface
  -> strict check on overlap
  -> estimate non-overlap valid area
  -> if non-overlap area is large and pose confidence high, accept as new surface
```

这会直接提高 TSDF 的有效融合帧数，也更符合“多视角补全”的目标。

### 2.4 先融合宽松 TSDF，再做 final quality rollback

为了提高成功率，可以把前置 gate 放松，让更多帧进入临时融合；最终再用更强的 geometry proxy 决定是否保留。

```text
permissive fusion
  -> temporary TSDF mesh
  -> geometry proxy evaluation
  -> if improved: keep
  -> else: rollback to previous best / base prior
```

geometry proxy 应更接近最终指标：

```text
observed-surface Chamfer
visible-region F-score
projection IoU
surface coverage
prior preservation penalty
```

关键点是：**前面提高成功率，后面用 rollback 保证不伤几何质量。**

### 2.5 使用 prior-guided visible-region update

不要让 TSDF mesh 直接替换 SAM3D prior。推荐做：

```text
base generative mesh
  -> compute face visibility / confidence
  -> TSDF reconstruct observed surface
  -> update only visible high-confidence region
  -> keep prior in unobserved region
  -> blend boundary between TSDF patch and prior
```

这样可以同时保留：

```text
generative prior 的完整性
TSDF 对真实可见表面的修正能力
```

这比“TSDF 输出完整 model.obj”更可能提升 Chamfer 和 F1@0.01。

### 2.6 对称 part 保留多 pose / alignment 候选

理想观测也不能消除几何对称性。对称 part 的多个 pose / axis alignment 都可能局部合理。

建议：

```text
axis alignment top-k
  -> seed pose top-k
  -> TSDF branch top-k
  -> final geometry proxy chooses best
```

不要在 OBB alignment 或 seed ICP 阶段过早只保留一个解。

## 3. BundleSDF 可借鉴内容

这里不借鉴它的噪声处理，而借鉴理想条件下仍然有用的工程思想。

### 3.1 Scene bounds normalization

BundleSDF 会根据多帧 object points 估计 `translation` 和 `sc_factor`，把优化空间归一化。TSDF 可以借鉴这一点：

```text
multi-frame part point cloud
  -> estimate stable center and scale
  -> normalize mesh and point cloud for registration
  -> run ICP / fusion in normalized space
  -> map result back to part frame
```

这能降低尺度差异和 center offset 对 ICP 的影响。

### 3.2 Pose array optimization

BundleSDF 不是完全相信初始 pose，而是在优化过程中维护 pose correction。TSDF 可以借鉴为：

```text
accepted frames
  -> joint optimize small pose deltas
  -> constrain max rotation / translation
  -> rebuild TSDF with optimized poses
```

这比每帧单独 ICP 更稳定，尤其适合 articulated parts 的局部观测。

### 3.3 Coarse-to-fine truncation schedule

BundleSDF 的 SDF truncation 有 coarse-to-fine 思路。TSDF 也可以：

```text
pass 1: larger voxel_length / sdf_trunc, easier to fuse
pass 2: smaller voxel_length / sdf_trunc, refine surface
```

第一阶段追求成功和连通，第二阶段追求精度。

### 3.4 Octree / occupied-space restriction

BundleSDF 用 octree 限制采样空间。TSDF 可以用更轻量的 occupied-space prior：

```text
base mesh bbox + observed point bbox
  -> define valid fusion volume
  -> ignore updates far outside this volume
```

这不是为了去噪，而是为了避免 TSDF 在不相关空间生成碎片或空表面。

## 4. Any6D 可借鉴内容

这里同样排除 depth/mask 噪声处理，只保留与理想观测仍相关的部分。

### 4.1 OBB axis alignment

Any6D 会用 observed point cloud OBB 与 mesh OBB 做 axis alignment。这个对理想观测仍然有价值，因为它解决的是坐标系/尺度初始化，而不是噪声。

建议：

```text
observed OBB + mesh OBB
  -> enumerate axis permutations and signs
  -> reject mirrored transforms
  -> keep top-k candidates
  -> TSDF seed branch chooses best
```

## 5. 推荐实现顺序

```text
Step 1: seed top-k branch selection
Step 2: information-gain view selection
Step 3: overlap consistency + new-surface utility
Step 4: relaxed gate + final geometry rollback
Step 5: BundleSDF-style scale / center normalization
Step 6: prior-guided visible-region TSDF update
Step 7: coarse-to-fine TSDF schedule
```

预期对成功率的提升排序：

```text
seed top-k
  > relaxed gate + rollback
  > information-gain view selection
  > scale / center normalization
  > overlap/new-surface split
```

预期对几何质量的提升排序：

```text
prior-guided visible-region update
  > final geometry rollback / best-of-N
  > pose/scale normalization
  > information-gain view selection
  > coarse-to-fine TSDF
```
