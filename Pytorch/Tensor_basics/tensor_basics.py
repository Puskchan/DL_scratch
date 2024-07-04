import torch
import numpy as np

device = torch.device("cuda:0")
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Common initialization methods
x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5) # Identity matrix
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1,end=1,steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
x = torch.diag(torch.ones(3))


# Conversion of tensors

tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())


# Array to tensor and vice versa
np_arr = np.zeros((5,5))
tensor = torch.from_numpy(np_arr)
print(tensor)
