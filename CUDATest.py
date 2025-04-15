import torch


# 验证CUDA是否可用打印版本版本
def check_cuda():
    print(f"PyTorch版本：{torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA可用！版本为：{torch.version.cuda}")
        print(f"CUDA设备名：{torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA不可用.")


if __name__ == "__main__":
    check_cuda()
