import torch

if __name__ == "__main__":
    N = 16
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = a + b
    print(a, b, c, sep="\n")