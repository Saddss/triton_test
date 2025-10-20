import torch
import triton
import triton.language as tl
import time

@triton.jit
def test_with_mod(input_ptr, output_ptr, M, BLOCK_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    offsets = (pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    mask = offsets < M
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)

@triton.jit  
def test_without_mod(input_ptr, output_ptr, M, BLOCK_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < M
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)

def profile(func, inputs, num_warmups=50, num_iters=50):
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func(*inputs)
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end) / num_iters
    return latency
    
def benchmark():
    M = 2048
    BLOCK_SIZE = 128
    input_tensor = torch.randn(M, device='cuda', dtype=torch.float16)
    output_mod = torch.zeros_like(input_tensor)
    output_no_mod = torch.zeros_like(input_tensor)
    
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    
    # 定义kernel调用函数
    def call_with_mod():
        test_with_mod[grid](input_tensor, output_mod, M, BLOCK_SIZE_M=BLOCK_SIZE)
    
    def call_without_mod():
        test_without_mod[grid](input_tensor, output_no_mod, M, BLOCK_SIZE_M=BLOCK_SIZE)
    
    # 使用profile函数测试取模版本
    mod_latency = profile(call_with_mod, (), num_warmups=50, num_iters=100)
    
    # 使用profile函数测试不取模版本
    no_mod_latency = profile(call_without_mod, (), num_warmups=50, num_iters=100)
    
    print(f"=== 性能测试结果 ===")
    print(f"取模版本平均耗时: {mod_latency:.4f}ms")
    print(f"不取模版本平均耗时: {no_mod_latency:.4f}ms")
    print(f"性能提升: {(mod_latency/no_mod_latency - 1)*100:.2f}%")
    print(f"结果是否相同: {torch.allclose(output_mod, output_no_mod)}")

if __name__ == '__main__':
    benchmark()
