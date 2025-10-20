import torch
import triton
import triton.language as tl

@triton.jit
def block_matmul_kernel(input_ptr, weight_ptr, output_ptr, M, N, K, stride_input_m, stride_input_k, stride_weight_k, stride_weight_n, stride_output_m, stride_output_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < K

        input_block = tl.load(input_ptr + offsets_m[:, None] * stride_input_m + offsets_k[None, :] * stride_input_k, mask=mask_m[:, None] & mask_k[None, :])
        weight_block = tl.load(weight_ptr + offsets_k[:, None] * stride_weight_k + offsets_n[None, :] * stride_weight_n, mask=mask_k[:, None] & mask_n[None, :])
        output_block = tl.dot(input_block, weight_block, acc=output_block)
    output = output_block.to(output_ptr.dtype.element_ty) # 显式指定类型转换，不指定也会隐式转换为tl.float16
    offsets_output = offsets_m[:, None] * stride_output_m + offsets_n[None, :] * stride_output_n
    tl.store(output_ptr + offsets_output, output_block, mask=mask_m[:, None] & mask_n[None, :])

def block_matmul_triton(input, weight):
    M, K_1 = input.shape
    K_2, N = weight.shape
    assert K_1 == K_2
    K = K_1
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), )
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']), )
    block_matmul_kernel[grid](
    input,
    weight,
    output,
    M, N, K,
    *input.stride(),
    *weight.stride(),
    *output.stride(),
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    num_warps=8, num_ctas=1, num_stages=3)
    return output

if __name__ == '__main__':
    input_dim = 2048
    hidden_dim = 4096
    output_dim = 2048

    input = torch.randn((input_dim, hidden_dim), device='cuda', dtype=torch.float16)
    weight = torch.randn((hidden_dim, output_dim), device='cuda', dtype=torch.float16)

    output = block_matmul_triton(input, weight)
    golden = input @ weight
    print(output.shape)  
    print(golden.shape)
    print(output)
    print(golden)
    print('两者是否接近： ', torch.allclose(output, golden, atol=1e-6))
