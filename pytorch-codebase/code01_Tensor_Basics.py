import torch

# Everything in PyTorch is based on Tensor operations.
# A tensor can have different dimensions, so it can be 1d (scalar),
# 2d (vector), or even 3d (matrix) and higher. Let's have a look how we can create a tensor in PyTorch.

# torch.empty(size): uninitiallized
x = torch.empty(1)  # scalar
print(x)
x = torch.empty(3)  # vector, 1D
print(x)
x = torch.empty(2, 3)  # matrix, 2D
print(x)
x = torch.empty(2, 2, 3)  # tensor, 3 dimensions
# x = torch.empty(2,2,2,3) # tensor, 4 dimensions
print(x)

# torch.rand(size): random numbers [0, 1]
x = torch.rand(5, 3)
print(x)

# torch.zeros(size), fill with 0
# torch.ones(size), fill with 1
x = torch.zeros(5, 3)
print(x)

# check size
print(x.size())

# check data type
print(x.dtype)

# specify types, float32 default
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)

# check type
print(x.dtype)

# construct from data (list format)
x = torch.tensor([5.5, 3])
print(x.size())
print(x)

# This will tell PyTorch that it will need to calculate the gradients for this tensor later in your optimization steps,
# i.e., this is a variable in your model that you want to optimize. This will be very important later in our training!
x = torch.empty(1,5, requires_grad=True) # requires_grad*

# Operation
y = torch.rand(2, 2)
x = torch.rand(2, 2)

# elementwise addition
z = x + y
# torch.add(x,y)

# in place addition, everythin with a trailing underscore is an inplace operation
# i.e. it will modify the variable
# y.add_(x)

# subtraction
z = x - y
z = torch.sub(x, y)

# multiplication
z = x * y
z = torch.mul(x,y)

# division
z = x / y
z = torch.div(x,y)

# Slicing
x = torch.rand(5,3)
print(x)
print(x[:, 0]) # all rows, column 0
print(x[1, :]) # row 1, all columns
print(x[1,1]) # element at 1, 1

# Get the actual value if only 1 element in your tensor
print(x[1,1].item())

# Reshape with torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())

# Converting numpy to torch is easy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))

# If the Tensor is on the CPU (not the GPU), both objects will share the same memory location,
# so changing one will also change the other:
a.add_(1)
print(a)
print(b)

# numpy to torch using .from_numpy
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# again be careful of the modifying
a += 1
print(a)
print(b)

# By default all tensors are created on the CPU, but you can also move them to the GPU (only if it's available )
if torch.cuda.is_available():
    device = torch.device("cuda")      # a CUDA device object
    y = torch.ones(5, device=device)  # directly create a tensor on GPU
    x = torch.ones(5)
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    # z = z.numpy() # not possible because numpy cannot handle GPU tenors
    # move to CPU again
    z.to("cpu")       # ``.to`` can also change dtype together!
    # z = z.numpy()

print(torch.cuda.is_available())