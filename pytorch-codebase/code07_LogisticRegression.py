# Logistic Regression
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer() # binary classification dataset
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features) # 569, 30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# scale
sc = StandardScaler() # Standardization
X_train = sc.fit_transform(X_train) # fit_transform: Fit to data, then transform it.
X_test = sc.transform(X_test) # transform: Perform standardization by centering and scaling.

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
class LogisticRegression(nn.Module):
    def __init__(self, input_features):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(input_features, 1) # output_features is set to 1.
    def forward(self, x):
        y_predicted = torch.sigmoid(self.lin(x))
        return y_predicted

model = LogisticRegression(input_features=n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # Binary Cross Entropy, 交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
n_iters = 100
for epoch in range(n_iters):
    # forward pass and loss
    y_pred = model(X_train)
    l = criterion(y_pred, y_train)

    # backward pass
    l.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: loss: {l:.3f}")

# eval
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() # if value > 0.5 =1, <0.5=0
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")