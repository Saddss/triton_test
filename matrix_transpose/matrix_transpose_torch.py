import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    # transposedmat = input.transpose(-1, -2)
    # transposedmat = input.t()
    
    # 高维度用这个 全部维度转置
    # transposedmat = input.T()
    transposedmat = input.permute(1, 0)
    output.copy_(transposedmat)
    pass 

if __name__ == '__main__':
    input = torch.randn((1, 2), device='cuda:0')
    output = torch.empty((2, 1), device=input.device)
    solve(input=input, output=output, rows=1, cols=2)
    print(input)
    print(output)