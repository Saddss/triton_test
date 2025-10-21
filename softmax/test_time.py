import torch

from softmax_triton import softmax_triton
from softmax_triton1 import softmax


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


def test_softmax(M: int, N: int):
    # 为每个测试创建独立的张量，避免内存冲突
    a1 = torch.randn((M, N), dtype=torch.float16, device='cuda', requires_grad=False)
    a2 = torch.randn((M, N), dtype=torch.float16, device='cuda', requires_grad=False)
    a3 = torch.randn((M, N), dtype=torch.float16, device='cuda', requires_grad=False)
    
    b = torch.empty((M, N), dtype=torch.float16, device='cuda', requires_grad=False)
    
    # 分别测试每个实现
    c_torch = torch.softmax(a1, dim=-1)
    torch.cuda.synchronize()  # 确保GPU操作完成
    
    c_triton_block = softmax_triton(a2, b, a2.stride(0), b.stride(0))
    torch.cuda.synchronize()  # 确保GPU操作完成
    
    c_triton = softmax(a3)
    torch.cuda.synchronize()  # 确保GPU操作完成

    # 验证正确性（使用相同的输入）
    test_a = torch.randn((M, N), dtype=torch.float32, device='cuda', requires_grad=False)
    torch_result = torch.softmax(test_a, dim=-1)
    torch.cuda.synchronize()
    
    triton_block_result = softmax_triton(test_a.clone(), torch.empty_like(test_a), test_a.stride(0), test_a.stride(0))
    torch.cuda.synchronize()
    
    triton_result = softmax(test_a.clone())
    torch.cuda.synchronize()

    # 性能测试 - 为每个测试创建新的张量
    latency_torch = profile(lambda x: torch.softmax(x, dim=-1), (a1,))
    latency_triton_block = profile(softmax_triton, (a2, b, a2.stride(0), b.stride(0)))
    latency_triton_group = profile(softmax, (a3,))

    print(f'Problem size: ({M}, {N})')
    print(f'Torch torch_softmax Latency: {latency_torch:.3f} ms')
    print(f'Triton triton_softmax_block Latency: {latency_triton_block:.3f} ms')
    print(f'Triton triton_softmax Latency: {latency_triton_group:.3f} ms')


if __name__ == '__main__':
    # 测试不同大小的矩阵
    sizes = [
        (2048, 4096),      # 小矩阵
        (4096, 16384),     # 中等矩阵  
        (4096, 65536),     # 大矩阵
        (4096, 131072)    # 更大矩阵
    ]
    
    for M, N in sizes:
        print(f"\n{'='*50}")
        try:
            test_softmax(M, N)
        except Exception as e:
            print(f"Error with size ({M}, {N}): {e}")
            break