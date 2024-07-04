import numpy as np
import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

#Addition
z1 = torch.add(x,y) #too verbose
z = x + y

#Subtraction
z = x - y

#Division
z = torch.true_divide(x,y)

#inplace operation
t = torch.zeros(3)
t.add_(x)

#Exponentiation
z = x.pow(2)
z = x ** 2

#Comparision
z = x>0
z = x<0

#Mat Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

#Mat exponent
mat_exp = torch.rand(5,5)
mat_exp.matrix_power(3)

#Element wise multiplication
z = x*y

#Dot product
z = torch.dot(x,y)


#Batch mat multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1, tensor2)


#Broadcasting

x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2 # This is not possible in real life
# We expand x2 to match x1 and then do the subtraction 
# also known as broadcasting.
z = x1 ** x2 # This is done element wise, not to the whole matrix

#Miscellaneous Ops
sum_x = torch.sum(x,dim=0) #x.sum(dim=0) works too, you can do that for all
values, indices = torch.max(x,dim=0)
values, indices = torch.min(x,dim=0)
abs_x = torch.abs(x)
z = torch.argmin(x,dim=0)
z = torch.argmax(x,dim=0)
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
sorted_y, indices = torch.sort(y,dim=0, descending=False)

z = torch.clamp(x, min=0)

# Bool
x = torch.tensor([1,0,1,1,1], dtype=bool)
print(x.any()) # or torch.any(x)
print(x.all()) # or torch.all(x)