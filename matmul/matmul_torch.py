import torch
if __name__ == '__main__':
    a = torch.tensor([1, 2])
    b = torch.tensor([[1],[2]])
    # output = a @ b
    # output = a.matmul(b)
    output = torch.matmul(a, b)
    print(output)