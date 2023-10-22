import numpy as np
import matplotlib.pyplot as plt

# Redefine the functions to ensure they are present in the current context

# Linear regression model
def predict(X, weights):
    return X.dot(weights)

def compute_loss(y_true, y_pred):
    m = len(y_true)
    return ((y_pred - y_true) ** 2).sum() / (2 * m)

def gradients(X, y, y_pred):
    m = len(y)
    return -1/m * X.T.dot(y - y_pred)

def gradient_descent(X, y, lr, epochs, init_weights):
    m = len(y)
    loss_history = []
    weights = init_weights.copy()
    for epoch in range(epochs):
        y_pred = predict(X, weights)
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)

        grads = gradients(X, y, y_pred)
        weights -= lr * grads

    return weights, loss_history

def momentum(X, y, lr, epochs,init_weights, gamma=0.9 ):
    m = len(y)
    loss_history = []
    weights = init_weights.copy()
    v = np.zeros_like(weights)
    for epoch in range(epochs):
        y_pred = predict(X, weights)
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)

        grads = gradients(X, y, y_pred)
        v = gamma * v + lr * grads
        weights -= v

    return weights, loss_history

# Train with a given optimizer
def train_with_optimizer(X, y, optimizer, epochs, init_weights):
    m, n = X.shape
    weights = init_weights.copy()
    loss_history = []
    for epoch in range(epochs):
        y_pred = predict(X, weights)
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)

        grads = gradients(X, y, y_pred)
        weight_update = optimizer.apply_gradients(grads)
        weights -= weight_update
    return weights, loss_history
# def train_with_modified_optimizer(X, y, optimizer, epochs, init_weights):
#     m, n = X.shape
#     weights = init_weights.copy()
#     loss_history = []
#     for epoch in range(epochs):
#         y_pred = predict(X, weights)
#         loss = compute_loss(y, y_pred)
#         loss_history.append(loss)

#         grads = gradients(X, y, y_pred)
#         weight_update = optimizer.apply_gradients(grads, weights)
#         weights -= weight_update
#     return weights, loss_history
# 2. Update X_b after modifying the synthetic data

# First synthetic data
X1 = 2 * np.random.rand(100, 1)
y1 = 4 + 100 * X1 + np.random.randn(100, 1)
X1_b = np.c_[np.ones((100, 1)), X1]
y_preds = np.linspace(y1.min(), y1.max(), 400)


X2 = 2 * np.random.rand(100, 1)
y2 = 4 + 50 * X2 + np.random.randn(100, 1)
X2_b = np.c_[np.ones((100, 1)), X2]
lr = 0.01

# 3. Properly store and plot the training histories for the two different datasets
from test_optimizer2 import TestOptimizer 

# Training on the first dataset
initial_weights1 = np.array([[3.], [8.]])
weights_gd_1, history_gd_1 = gradient_descent(X1_b, y1, lr=lr, epochs=2000, init_weights=initial_weights1)
power_optimizer1 = TestOptimizer(alpha=lr, beta=0.9,b=2)
_, history_power_optimizer_1 = train_with_optimizer(X1_b, y1, power_optimizer1, epochs=2000, init_weights=initial_weights1)
power_optimizer2 = TestOptimizer(alpha=lr, beta=0.9,b=1.1)
_, history_power_optimizer_2 = train_with_optimizer(X1_b, y1, power_optimizer1, epochs=2000, init_weights=initial_weights1)
power_optimizer3 = TestOptimizer(alpha=lr, beta=0.9,b=0.9)
_, history_power_optimizer_3 = train_with_optimizer(X1_b, y1, power_optimizer1, epochs=2000, init_weights=initial_weights1)
power_optimizer4 = TestOptimizer(alpha=lr, beta=0.9,b=0.)
_, history_power_optimizer_4 = train_with_optimizer(X1_b, y1, power_optimizer1, epochs=2000, init_weights=initial_weights1)
# Training on the second dataset
# initial_weights2 = np.array([[11.], [44.]])
# power_optimizer2 = TestOptimizer(alpha=lr, beta=0.99)
# po2w, history_power_optimizer_2 = train_with_optimizer(X2_b, y2, power_optimizer2, epochs=1000, init_weights=initial_weights2)

# Plotting the results for the two datasets

# First dataset
plt.figure(figsize=(10, 6))

# plt.plot(history_power_optimizer_1, label='PowerOptimizer1')
plt.plot(history_gd_1, label='Gradient Descent')
plt.plot(history_power_optimizer_2, label='PowerOptimizer2')
plt.plot(history_power_optimizer_3, label='PowerOptimizer3')
plt.plot(history_power_optimizer_4, label='PowerOptimizer4')

plt.xlim(0, 2000)
plt.ylim(0, 100)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimizer Comparison on First Dataset')
plt.show()

# # Second dataset
# plt.figure(figsize=(10, 6))
# # plt.plot(history_gd_2, label='Gradient Descent')
# # plt.plot(history_momentum_2, label='Momentum')
# plt.plot(history_power_optimizer_2, label='PowerOptimizer')
# # plt.xlim(25, 80)
# # plt.ylim(0, 100)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Optimizer Comparison on Second Dataset')
# plt.show()

