import torch

batch_size = 10
features = 25
x = torch.rand((batch_size,features))

print(x[0,:]) # first feature full

print(x[:,0]) # first feature over all the examples

print(x[2,:10]) # 3rd row,10 examples

x[0,0] = 100 # Assign value

#Fancy indices
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
r = torch.tensor([1,0])
c = torch.tensor([4,0])
print(x[r,c])


#Advance Indexing
x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[x.remainder(2)==0])

# Other Ops
print(torch.where(x>5, x, x*2))

print(torch.tensor([0,0,1,2,2,3,4]).unique())

print(x.ndimension())

print(x.numel())