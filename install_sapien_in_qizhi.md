
### ManiSkill (written by zehao)

Install

```bash
# apt
apt-get install libglvnd-dev libvulkan1 vulkan-tools -y

# add one files
cat /etc/vulkan/implicit_layer.d/nvidia_layers.json | grep api_version (1.4.303)
vim /usr/share/vulkan/icd.d/nvidia_icd.json
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.277"   # need change
    }
}

# python
pip install --upgrade mani_skill

# environment variables
export MS_ASSET_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/.dataset/maniskill
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=0
```

# comment by yixuan

- 执行`vim /usr/share/vulkan/icd.d/nvidia_icd.json`的时候不要把 带“#”的注释也写进去了

- 可参考 https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan
- export MS_ASSET_DIR这里要导入自己的maniskill路径

常用命令
```bash
vulkaninfo --summary
apt install vulkan-tools
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

# comment by yuquan

注意开镜像时，镜像的cuda版本尽量和宿主机的cuda版本一致，否则可能会有libglx报错

    ERROR: [Loader Message] Code 0 : loader_scanned_icd_add: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0