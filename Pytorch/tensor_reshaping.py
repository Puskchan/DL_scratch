import torch

x = torch.arange(9)

x_3 = x.view(3,3)
print(x_3)
x_3_r = x.reshape(3,3)
print(x_3_r)
# Both do the same work, but for view we need a contigous elements
# but for reshape it just makes a copy
# view is faster but reshape is safer

y = x_3.t()
print(y)
# now the view wont work with this
#        print(y.view(9)) 
# trying to convert the back to 1d array (gives an error)

# but you could do this!
print(y.contiguous().view(9)) # and this works, for obvious reasons
# Beware of this when writing code or just use reshape


x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=1))
print(torch.cat((x1,x2), dim=0))
print(torch.cat((x1,x2), dim=0).shape)
print(torch.cat((x1,x2), dim=1).shape)


z = x1.view(-1)
print(z)

batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)


z = x.permute(0,2,1)
print(z.shape)

x = torch.arange(10)
print(x)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x.shape)
z = x.squeeze(1)
print(z.shape)