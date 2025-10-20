import torch
import triton
import triton.language as tl

@triton.jit
def block_matmul_kernel(input_ptr, weight_ptr, output_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        input_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        input = tl.load(input_ptr + offset_m * K + input_k, mask=(offset_m < M) & (input_k < K))

        weight_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        weight = tl.load(weight_ptr + weight_k * N + offset_n, mask=(weight_k < K) & (offset_n < N))

        output = tl.dot(input, weight, acc=output)
    offset_output = offset_m * N + offset_n
    tl.store(output_ptr + offset_output, output, mask=(offset_m < M) & (offset_n < N))

def block_matmul_triton(input, weight):
    M, K_1 = input.shape
    K_2, N = weight.shape
    assert K_1 == K_2
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    block_matmul_kernel[grid](
    input,
    weight,
    output,
    M, N, K_1,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return output

if __name__ == '__main__':
    input_dim = 128
    hidden_dim = 128
    output_dim = 256

    input = torch.randn((input_dim, hidden_dim), device='cuda', dtype=torch.float16)
    weight = torch.randn((hidden_dim, output_dim), device='cuda', dtype=torch.float16)

    output = block_matmul(input, weight)
    golden = input @ weight
    print(output.shape)  
    print(golden.shape)
    print(output)
    print(golden)
    print('两者是否接近： ', torch.allclose(output, golden, atol=1e-6))
