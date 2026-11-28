import torch

w = torch.tensor(3.0,requires_grad=True)

y = w**2 + w*2

y.backward()
print("first backward: ",w.grad.item())

y2 = w**2 + w*2
y2.backward()
print("without zero grad: ",w.grad.item())

w.grad.zero_()
print("after zero grad: ",w.grad.item())