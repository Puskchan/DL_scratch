import torch

# Creating a tensor with random values of shape (batch_size, features)
batch_size = 10  # Number of samples in the batch
features = 25  # Number of features per sample
x = torch.rand((batch_size, features))  # Generates random values between 0 and 1

# Indexing operations on tensors
print(x[0, :])  # Prints all features of the first sample (first row)

print(x[:, 0])  # Prints the first feature across all samples (first column)

print(x[2, :10])  # Prints the first 10 features of the 3rd sample (row index 2)

# Assigning a new value to a specific element in the tensor
x[0, 0] = 100  # Sets the first element in the first row to 100

# Fancy Indexing (Selecting specific elements using a list of indices)
x = torch.arange(10)  # Creates a tensor with values from 0 to 9
indices = [2, 5, 8]  # List of indices to select specific elements
print(x[indices])  # Prints elements at indices 2, 5, and 8

# Advanced Indexing with Multiple Indices
x = torch.rand((3, 5))  # Creates a 3x5 matrix with random values
r = torch.tensor([1, 0])  # Row indices
c = torch.tensor([4, 0])  # Column indices
print(x[r, c])  # Selects elements at (1,4) and (0,0)

# Boolean Masking & Conditional Indexing
x = torch.arange(10)  # Tensor with values 0 to 9
print(x[(x < 2) | (x > 8)])  # Selects elements that are either <2 or >8
print(x[x.remainder(2) == 0])  # Selects only even numbers

# Other Tensor Operations
# torch.where(condition, value_if_true, value_if_false)
print(torch.where(x > 5, x, x * 2))  # If x > 5, keep x; otherwise, multiply by 2

# Finding Unique Elements
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())  # Returns unique elements

# Getting Tensor Dimensions
print(x.ndimension())  # Prints number of dimensions (rank) of tensor

# Counting Total Elements in Tensor
print(x.numel())  # Returns total number of elements in tensor