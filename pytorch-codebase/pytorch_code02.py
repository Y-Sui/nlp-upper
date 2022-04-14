# Gradient Calculation With Autograd
import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y) # return the add-backward function output
z = y*y*2
print(z) # return the mul-backward function output
# z = z.mean()
# print(z)
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) # dz/dx
print(x.grad)

# 3 options to stop auto-grad
# x.requires_grad(False)
# x.detach()
# with torch.no_grad():

x = torch.randn(3, requires_grad=True)
y = x.detach()
print(x)
print(y)

weights = torch.ones(4, requires_grad=True)
print(weights)
for epoch in range(3):
    model_output = (weights*3).sum()
    print(model_output)
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() # restart/empty the weights again

# optimizer
optimizer =torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()
