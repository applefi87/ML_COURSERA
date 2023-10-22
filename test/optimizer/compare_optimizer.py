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

# Compute the cost for each predicted value
# costs = [compute_loss(y1, y_pred * np.ones_like(y1)) for y_pred in y_preds]
# # Plot
# plt.figure(figsize=(10, 6))
# plt.plot(y_preds, costs, '-r', label='Cost Function')
# plt.xlabel('Predicted Value')
# plt.ylabel('Cost')
# plt.title('Cost Function Visualization')
# plt.legend()
# plt.grid(True)
# plt.show()
# Second synthetic data (ravine-like loss landscape)
X2 = 2 * np.random.rand(100, 1)
y2 = 4 + 50 * X2 + np.random.randn(100, 1)
X2_b = np.c_[np.ones((100, 1)), X2]
lr = 1
mlr = 0.1
# 3. Properly store and plot the training histories for the two different datasets
from test_optimizer2 import TestOptimizer 
from Adam_optimizer import AdamOptimizer
# Training on the first dataset
initial_weights1 = np.array([[3.], [8.]])
weights_gd_1, history_gd_1 = gradient_descent(X1_b, y1, lr=lr, epochs=1000, init_weights=initial_weights1)
weights_momentum_1, history_momentum_1 = momentum(X1_b, y1, lr=mlr, epochs=1000, init_weights=initial_weights1)
power_optimizer1 = TestOptimizer(alpha=lr, beta=0.99)
_, history_power_optimizer_1 = train_with_optimizer(X1_b, y1, power_optimizer1, epochs=1000, init_weights=initial_weights1)
adam_optimizer = AdamOptimizer(learning_rate=0.001)
_, history_adam_optimizer_1 = train_with_optimizer(X1_b, y1, adam_optimizer, epochs=1000, init_weights=initial_weights1)

# Training on the second dataset
initial_weights2 = np.array([[11.], [44.]])
weights_gd_2, history_gd_2 = gradient_descent(X2_b, y2, lr=lr, epochs=1000, init_weights=initial_weights2)
weights_momentum_2, history_momentum_2 = momentum(X2_b, y2, lr=mlr, epochs=1000, init_weights=initial_weights2)
power_optimizer2 = TestOptimizer(alpha=lr, beta=0.99)
po2w, history_power_optimizer_2 = train_with_optimizer(X2_b, y2, power_optimizer2, epochs=1000, init_weights=initial_weights2)

# Plotting the results for the two datasets

# First dataset
plt.figure(figsize=(10, 6))
# plt.plot(history_gd_1, label='Gradient Descent')
# plt.plot(history_momentum_1, label='Momentum')
plt.plot(history_power_optimizer_1, label='PowerOptimizer')
# plt.plot(history_adam_optimizer_1, label='AdamOptimizer')

# plt.xlim(20, 60)
# plt.ylim(0, 50)
# plt.xlim(100, 200)
# plt.ylim(0, 100)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimizer Comparison on First Dataset')
plt.show()

# Second dataset
plt.figure(figsize=(10, 6))
# plt.plot(history_gd_2, label='Gradient Descent')
# plt.plot(history_momentum_2, label='Momentum')
plt.plot(history_power_optimizer_2, label='PowerOptimizer')
# plt.xlim(25, 80)
# plt.ylim(0, 100)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimizer Comparison on Second Dataset')
plt.show()

print(po2w)