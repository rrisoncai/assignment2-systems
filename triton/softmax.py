import torch
import triton
import triton.language as tl

DEVICE = torch.cuda.current_device()
print(DEVICE)
