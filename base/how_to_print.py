import triton.language as tl
import triton
import torch

@triton.jit
def debug_kernel(X_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    
    # 打印线程块 ID (pid)
    tl.device_print("Current PID:", pid) 
    
    # 加载一个值
    x_val = tl.load(X_ptr + pid)
    
    # 打印加载的值
    tl.device_print("Value:", x_val) 
    
    # 打印一个向量/Tile (所有线程都会打印自己的值)
    vector = tl.arange(0, 4)
    tl.device_print("Vector:", vector)

if __name__ == '__main__':
    debug_kernel[(1,)](torch.tensor([1], device='cuda:0'), 1, num_warps=1)