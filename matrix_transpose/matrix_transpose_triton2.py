import torch
import triton.language as tl
import triton

@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    old_index = pid_row * cols + pid_col
    val = tl.load(input_ptr + old_index)

    transposed_index = pid_col * rows + pid_row

    tl.store(output_ptr + transposed_index, val)

def matrix_transpose_triton(input, output, rows, cols):
    grid = (rows, cols)
    matrix_transpose_kernel[grid](input, output, rows, cols)

if __name__ == '__main__':
    input = torch.randn((3, 2), device='cuda:0')
    output = torch.empty((input.shape[1], input.shape[0]), device=input.device)
    output_torch = input.t()
    matrix_transpose_triton(input, output, *input.shape)
    if torch.allclose(output, output_torch):
        print(True)
    else:
        print(False)