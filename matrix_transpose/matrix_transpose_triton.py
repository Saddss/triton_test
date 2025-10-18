import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input, output,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc,
    BLOCK_SIZE: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offsets = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    old_offsets = row_offsets[:, None] * stride_ir + col_offsets[None, :] * stride_ic

    input_mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    output_mask = (col_offsets[:, None] < cols) & (row_offsets[None, :] < rows)

    block = tl.load(input + old_offsets, mask=input_mask)

    # transposed_block = tl.trans(block)
    transposed_block = block.T

    transposed_offsets = col_offsets[:, None] * stride_or + row_offsets[None, :] * stride_oc

    tl.store(output + transposed_offsets, transposed_block, mask=output_mask)
    

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_SIZE = 32
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_SIZE=BLOCK_SIZE
    ) 

if __name__ == '__main__':
    input = torch.randn((1024, 1024), device='cuda:0')
    output = torch.empty((input.shape[1], input.shape[0]), device=input.device)
    output_torch = input.t()
    solve(input, output, *input.shape)
    if torch.allclose(output_torch, output):
        print(True)
    else:
        print(False)