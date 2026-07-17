# 重建环境配置
## 1. 一般不能用requirements.txt直接下载的包
1. pytorch3d
2. nvdiffrast
3. diff-gaussian-rasterization

需要用源码clone然后编译。其中diff-gaussian-rasterization从下面这个仓库来clone: https://github.com/autonomousvision/mip-splatting/tree/main/submodules/diff-gaussian-rasterization

这个库是用来做gaussian-splatting的

然后在这几个目录下pip install -e . --no-build-isolation