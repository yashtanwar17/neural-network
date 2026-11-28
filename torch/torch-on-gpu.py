import torch

device = torch.device("cuda")

a = torch.tensor([10.0, 20.0]).to(device)

b = torch.tensor([5.0, 2.0]).to(device)

c=a+b

print("Result:", c)

plain_list = c.cpu().tolist()
print("Back on CPU:", plain_list)