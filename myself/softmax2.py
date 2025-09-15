import torch
import triton
import triton.language as tl
  
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_len = 2
    row_start = tl.program_id(0) * row_len
    if row_start >= n_rows:
        return
    for row_idx in range(row_start, row_start + row_len, 1):
        # 定位一行的起点
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        # 表示某行中的所有元素下标
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        # 加载一行中的所有元素
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
           
        # 计算
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        # 为了写回定位一行的起点
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        # 写回
        tl.store(output_ptrs, softmax_output, mask=mask)


input_tensor = torch.randn(1000, 512, device='cuda')  # 1000x512 的随机张量
output_tensor = torch.empty_like(input_tensor)  # 用于存储输出的张量

# 2. 定义 kernel 的网格和块大小
n_rows, n_cols = input_tensor.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 计算大于 n_cols 的下一个 2 的幂
num_stages = 3  # 可以根据硬件调整

# 3. 调用 kernel
grid = lambda meta: (triton.cdiv(n_rows, 2),)  # 网格大小为行数
softmax_kernel[grid](
    output_tensor, input_tensor,
    input_tensor.stride(0), output_tensor.stride(0),  # 行步长
    n_rows, n_cols,
    BLOCK_SIZE=BLOCK_SIZE,
)

# 4. 验证结果
# 使用 PyTorch 的 softmax 作为参考
expected_output = torch.softmax(input_tensor, dim=1)
print("Triton Softmax 和 PyTorch Softmax 是否接近:", torch.allclose(output_tensor, expected_output, atol=1e-6))
