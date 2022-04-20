# Gradient Descent with Autograd and Backpropagation
#     Step1: Prediction: Manually -> Pytorch Model
#     Step2: Gradients Computation: Manually -> Autograd
#     Step3: Loss Computation: Manually -> Pytorch Loss
#     Step4: Parameter Updates: Manually -> Pytorch Optimizer

# ---------------------------------
# ------------Numpy----------------
# ---------------------------------
import numpy as np

# f = 2 * x
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x -y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 30

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f"Prediction after training: f(5) = {forward(5):.3f}")

# ---------------------------------
# ------------torch----------------
# ---------------------------------
import torch

# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32) # Input
Y = torch.tensor([2,4,6,8], dtype=torch.float32) # Ground Truth

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # Training Parameters

# model prediction
def forward(x):
    return w * x # Process the input and output the prediction

# loss
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean() # Define the loss function

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradient = backward pass
    l.backward() # dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f"Prediction after training: f(5) = {forward(5):.3f}")
