import torch

# Creating a 1D tensor with values from 0 to 8
x = torch.arange(9)

# Reshaping x into a 3x3 matrix using .view()
x_3 = x.view(3, 3)
print(x_3)

# Reshaping x into a 3x3 matrix using .reshape()
x_3_r = x.reshape(3, 3)
print(x_3_r)

# Difference between view() and reshape():
# - view() requires the tensor to have contiguous memory allocation.
# - reshape() can work even if the tensor is non-contiguous by making a copy if necessary.
# - view() is faster but less flexible, whereas reshape() is safer.

# Transposing the matrix (switching rows and columns)
y = x_3.t()
print(y)

# Attempting to use view() on a transposed tensor causes an error
# print(y.view(9))  # Uncommenting this will raise an error because y is non-contiguous

# Making the tensor contiguous before using view()
print(y.contiguous().view(9))
# The contiguous() method ensures the tensor has a continuous memory layout before applying view().
# Best practice: If unsure, use reshape() instead of view() to avoid such issues.

# Creating two random 2x5 tensors
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

# Concatenating tensors along different dimensions
print(torch.cat((x1, x2), dim=1))  # Concatenates along columns (axis 1)
print(torch.cat((x1, x2), dim=0))  # Concatenates along rows (axis 0)

# Checking the shape of the concatenated tensors
print(torch.cat((x1, x2), dim=0).shape)  # Should be (4,5) after stacking rows
print(torch.cat((x1, x2), dim=1).shape)  # Should be (2,10) after stacking columns

# Flattening a tensor using view()
z = x1.view(-1)  # -1 automatically infers the shape needed to flatten the tensor
print(z)

# Example of batch processing in neural networks
batch = 64  # Assume we have a batch of 64 samples
x = torch.rand((batch, 2, 5))  # Creating a batch of 64 tensors of shape (2,5)

# Flattening last two dimensions while keeping batch size
z = x.view(batch, -1)
print(z.shape)  # Output shape should be (64, 10)

# Changing dimension order using permute()
z = x.permute(0, 2, 1)  # Swaps dimensions 1 and 2
print(z.shape)  # Output should be (64, 5, 2)

# Unsqueezing tensors (adding a new dimension)
x = torch.arange(10)  # 1D tensor with shape (10,)
print(x)
print(x.unsqueeze(0).shape)  # Adds a new dimension at position 0 -> (1,10)
print(x.unsqueeze(1).shape)  # Adds a new dimension at position 1 -> (10,1)

# Stacking multiple unsqueeze operations
x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x.shape)  # Shape should be (1,1,10)

# Removing a dimension using squeeze()
z = x.squeeze(1)  # Removes dimension 1 if it's 1
print(z.shape)  # Output should be (1,10)