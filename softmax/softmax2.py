import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_ROWS: tl.constexpr,  # 块内处理的行数
    BLOCK_COLS: tl.constexpr   # 块内处理的列数（2的幂）
):
    # 网格级分片：获取当前Block的起始行和步长
    block_id = tl.program_id(0)
    row_start = block_id * BLOCK_ROWS
    rows = row_start + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < n_rows  # 过滤无效行

    # 列循环：处理所有列（修复列数超过BLOCK_COLS的问题）
    col_start = 0
    while tl.loop(col_start < n_cols):
        cols = col_start + tl.arange(0, BLOCK_COLS)
        col_mask = cols < n_cols  # 过滤无效列

        # 加载数据到寄存器（全局内存→寄存器）
        input_ptrs = input_ptr + rows[:, None] * input_row_stride + cols[None, :]
        x = tl.load(
            input_ptrs,
            mask=row_mask[:, None] & col_mask[None, :],
            other=-float('inf')
        )

        # 计算当前块的最大值（用于数值稳定）
        row_max = tl.max(x, axis=1)
        x -= row_max[:, None]

        # 计算指数
        x_exp = tl.exp(x)

        # 计算当前块的和
        x_sum = tl.sum(x_exp, axis=1)

        # 存储中间结果到输出
        output_ptrs = output_ptr + rows[:, None] * output_row_stride + cols[None, :]
        tl.store(
            output_ptrs,
            x_exp / x_sum[:, None],
            mask=row_mask[:, None] & col_mask[None, :]
        )

        # 移动到下一列块
        col_start += BLOCK_COLS


def softmax(x):
    # 输入检查
    assert x.ndim == 2, "输入必须是2D张量 [n_rows, n_cols]"
    assert x.device.type == 'cuda', "输入必须在GPU上"
    assert x.dtype in [torch.float32, torch.float16], "仅支持float32/float16"
    
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    # 动态调整块大小以获得更好的性能
    BLOCK_ROWS = 32  # 增加块行数以提高缓存利用率
    BLOCK_COLS = 256  # 使用固定的较大块列数，通过循环处理所有列
    
    # 计算网格大小
    grid_rows = triton.cdiv(n_rows, BLOCK_ROWS)
    
    # 配置Warp数量
    num_threads = BLOCK_ROWS * BLOCK_COLS
    num_warps = triton.cdiv(num_threads, 32)
    num_warps = min(num_warps, 32)  # 单Block最大32个Warp

    # 调用内核
    softmax_kernel[(grid_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=num_warps
    )

    return y


# 测试
if __name__ == "__main__":
    # 测试标准情况
    x = torch.randn((100_000, 1024), device='cuda', dtype=torch.float32)
    triton_out = softmax(x)
    torch_out = torch.softmax(x, dim=1)
    
    # 验证正确性
    max_error = torch.max(torch.abs(triton_out - torch_out))
    print(f"最大误差: {max_error.item():.6f}")
    print(f"前5行求和: {triton_out.sum(dim=1)[:5].tolist()}")
    
    # 测试列数超过BLOCK_COLS的情况
    x_large = torch.randn((100, 2048), device='cuda', dtype=torch.float32)
    triton_large = softmax(x_large)
    torch_large = torch.softmax(x_large, dim=1)
    max_error_large = torch.max(torch.abs(triton_large - torch_large))
    print(f"大列数情况最大误差: {max_error_large.item():.6f}")
    
    # 测试极端值情况
    x_extreme = torch.full((10, 5), -1e9, device='cuda')
    triton_extreme = softmax(x_extreme)
    print(f"极端值情况输出: {triton_extreme[0].tolist()}")
