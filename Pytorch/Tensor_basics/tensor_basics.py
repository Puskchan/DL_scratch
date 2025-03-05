import torch  # Importing the PyTorch library
import numpy as np  # Importing NumPy for array manipulations

# Setting up the device for computation (CUDA if available, else CPU)
device = torch.device("cuda:0")  # Assigns the first GPU (if available) as the device

# Creating a tensor with specified values
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]],  # A 2x3 matrix
    dtype=torch.float32,  # Setting data type as 32-bit floating point
    device=device,  # Storing the tensor on the GPU
    requires_grad=True  # Enabling automatic differentiation for gradient computation
)

# Printing the tensor and its properties
print(my_tensor)  # Displays the tensor values
print(my_tensor.dtype)  # Prints the data type of the tensor
print(my_tensor.device)  # Prints the device on which the tensor is stored (GPU or CPU)
print(my_tensor.shape)  # Prints the shape of the tensor (rows, columns)
print(my_tensor.requires_grad)  # Prints whether gradient computation is enabled

# Common methods to initialize tensors
x = torch.empty(size=(3,3))  # Creates an uninitialized 3x3 tensor (values are random memory)
x = torch.zeros((3,3))  # Creates a 3x3 tensor filled with zeros
x = torch.rand((3,3))  # Creates a 3x3 tensor with random values between 0 and 1
x = torch.ones((3,3))  # Creates a 3x3 tensor filled with ones
x = torch.eye(5,5)  # Creates a 5x5 identity matrix (diagonal elements are 1, others 0)
x = torch.arange(start=0, end=5, step=1)  # Creates a tensor with values from 0 to 4 (step of 1)
x = torch.linspace(start=0.1, end=1, steps=10)  # Creates 10 evenly spaced values between 0.1 and 1
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)  # Creates a 1x5 tensor with values drawn from a normal distribution (mean=0, std=1)
x = torch.diag(torch.ones(3))  # Creates a 3x3 diagonal matrix with ones on the diagonal

# Conversion between different tensor data types
tensor = torch.arange(4)  # Creates a tensor with values [0, 1, 2, 3]
print(tensor.bool())  # Converts tensor to boolean type (non-zero values become True, zero remains False)
print(tensor.short())  # Converts tensor to 16-bit integer (int16)
print(tensor.long())  # Converts tensor to 64-bit integer (int64, default integer type in PyTorch)
print(tensor.half())  # Converts tensor to 16-bit floating point (float16, used for reduced precision computation)
print(tensor.float())  # Converts tensor to 32-bit floating point (float32, default float type in PyTorch)
print(tensor.double())  # Converts tensor to 64-bit floating point (float64, higher precision)

# Converting between NumPy arrays and PyTorch tensors
np_arr = np.zeros((5,5))  # Creates a 5x5 NumPy array filled with zeros
tensor = torch.from_numpy(np_arr)  # Converts NumPy array to a PyTorch tensor
print(tensor)  # Displays the converted tensor