import numpy as np
import torch

# Creating two simple tensors
x = torch.tensor([1,2,3])  # A 1D tensor with elements [1,2,3]
y = torch.tensor([9,8,7])  # A 1D tensor with elements [9,8,7]

# Addition
z1 = torch.add(x, y)  # Adds corresponding elements of x and y explicitly
z = x + y  # Performs element-wise addition using shorthand

# Subtraction
z = x - y  # Element-wise subtraction

# Division
z = torch.true_divide(x, y)  # Element-wise division ensuring float output

# In-place operation (modifies the tensor in memory, saves computation)
t = torch.zeros(3)  # Creates a tensor of zeros with shape (3,)
t.add_(x)  # In-place addition (modifies t instead of creating a new tensor)

# Exponentiation
z = x.pow(2)  # Raises each element in x to the power of 2
z = x ** 2  # Equivalent shorthand for element-wise exponentiation

# Comparison operations
z = x > 0  # Element-wise comparison, returns tensor([True, True, True])
z = x < 0  # Element-wise comparison, returns tensor([False, False, False])

# Matrix multiplication
x1 = torch.rand((2,5))  # Creates a 2x5 matrix with random values
x2 = torch.rand((5,3))  # Creates a 5x3 matrix with random values
x3 = torch.mm(x1, x2)  # Matrix multiplication of (2,5) * (5,3) -> (2,3)
x3 = x1.mm(x2)  # Alternative notation for matrix multiplication

# Matrix exponentiation
mat_exp = torch.rand(5,5)  # Creates a 5x5 matrix with random values
mat_exp.matrix_power(3)  # Raises the matrix to the power of 3 (mat * mat * mat)

# Element-wise multiplication
z = x * y  # Multiplies corresponding elements of x and y

# Dot product (inner product of two vectors)
z = torch.dot(x, y)  # Computes sum of (x[i] * y[i])

# Batch matrix multiplication
batch = 32  # Number of matrices in the batch
n = 10  # Number of rows in first set of matrices
m = 20  # Number of columns in first set and rows in second set
p = 30  # Number of columns in second set of matrices

tensor1 = torch.rand((batch, n, m))  # Batch of (10x20) matrices
tensor2 = torch.rand((batch, m, p))  # Batch of (20x30) matrices
out_bmm = torch.bmm(tensor1, tensor2)  # Batch-wise matrix multiplication -> (batch, n, p)

# Broadcasting (Expanding smaller tensor to match larger tensor)
x1 = torch.rand((5,5))  # 5x5 matrix
x2 = torch.rand((1,5))  # 1x5 matrix

z = x1 - x2  # x2 is expanded to (5,5) automatically before subtraction
z = x1 ** x2  # Element-wise exponentiation

# Miscellaneous operations
sum_x = torch.sum(x, dim=0)  # Sum of all elements in x
values, indices = torch.max(x, dim=0)  # Maximum value and its index in x
values, indices = torch.min(x, dim=0)  # Minimum value and its index in x
abs_x = torch.abs(x)  # Absolute values of x
z = torch.argmin(x, dim=0)  # Index of minimum element in x
z = torch.argmax(x, dim=0)  # Index of maximum element in x
mean_x = torch.mean(x.float(), dim=0)  # Mean of elements in x (converted to float for accuracy)
z = torch.eq(x, y)  # Checks if corresponding elements of x and y are equal
sorted_y, indices = torch.sort(y, dim=0, descending=False)  # Sort y in ascending order

# Clamping (limiting values within a range)
z = torch.clamp(x, min=0)  # Clamps all negative values to 0

# Boolean operations
x = torch.tensor([1,0,1,1,1], dtype=bool)  # Boolean tensor
print(x.any())  # Checks if any value is True (1)
print(x.all())  # Checks if all values are True (1)