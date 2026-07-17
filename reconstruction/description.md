Part-aware RGB-D Feature Lifting
              ↓
Context-conditioned Part Reconstruction
              ↓
Kinematic Graph-guided Pose and Assembly Optimization


# 模块功能
从reference帧的RGBD以及eccv模块提供的reference帧的part掩码，利用SAM3D/hunyuan3D/instantmesh重建一个mesh先验。然后结合inference帧的观测信息用TSDF对这个先验进行优化，补全reference看不到部分的几何结构。然后再用nvdiffrast中的dlmesh对这个模型做调优。最后得到一个优化后pose-utility的模型和一个inference帧的coarse pose。用于下游位姿估计。

# 解决的问题
1. single reference提供的观测信息有限。而下游位姿估计需要一个比较完整的mesh
2. 看不到的角度存在不确定性，可能会误导pose错，需要用inference中的信息尽可能补充。
3. 重建可能存在几何上的误差，需要进行尽可能的修正，使之更贴近真实几何。
4. part和整体的关系需要建模，需要规避单独看一个part mesh位姿没问题，然后放到整个物体下就会因为这个part本身对称发生若干度的旋转这种情况。

# 输入
1. reference帧的RGBD，以及上游模块得到的part mask。
2. inference帧的RGBD，以及part mask。
3. 相机的内参。

# 输出
1. part的重建mesh。
2. 一个part relation graph，描述part和整体的关系。


# 目前主pipeline构成

    SAM3D base
    -> reference-only PartCut
    -> Any6D-inspired axis alignment
    -> TSDF refinement
    -> DLMesh refinement
    -> relation graph export