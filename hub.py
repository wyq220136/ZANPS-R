from huggingface_hub import login, HfApi

# 建议直接在 login 中填入 token，或者先在终端执行 huggingface-cli login
login() 

api = HfApi()

# 上传单个大文件
api.upload_file(
    path_or_fileobj=r"D:\research\PartNet\reconstruct.tar.gz",
    path_in_repo="reconstruct.tar.gz",
    repo_id="XiuluoWang/partnet_code",
    repo_type="dataset"
)