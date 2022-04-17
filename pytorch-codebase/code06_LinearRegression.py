# Linear Regression
# 1) Design model (input, output size, forward pass)
# 2) Construct the loss and optimizer, torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=learning_rate)
# 3) Training loop
#    - forward pass: compute prediction
#    - backward pass: gradients
#    - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1) # reshape Y from flatten to one column

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = n_features
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_dim=input_size, output_dim=output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # Mean squared error 最小二乘
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
n_iters = 1000
for epoch in range(n_iters):
    # forward pass and loss
    y_pred = model(X)
    l = criterion(Y, y_pred)

    # backward pass
    l.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: loss: {l:.3f}")

# plot
predicted = model(X).detach().numpy() # prevent this from the computation graph using .detach()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()