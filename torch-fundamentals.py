import torch

device = torch.device("cuda")

grid = torch.tensor([
    [1.0,2.0,3.0],
    [4.0,5.0,6.0]
])

zeros = torch.zeros((2,3))
rand = torch.randn((2,3))

print("Shape : ",zeros.shape)
print("Type : ",grid.dtype)

flat = grid.reshape(6)
row = flat.unsqueeze(0)
col = flat.unsqueeze(1)

print("Flat:", flat.shape, "| Column:", col.shape, "| Row:", row.shape)

