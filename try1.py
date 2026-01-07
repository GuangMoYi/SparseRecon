import torch
print(torch.__version__)          # 应输出类似 2.3.0+cu122
print(torch.cuda.is_available())  # 应返回 True
print(torch.version.cuda)         # 应返回 12.2