import numpy as np
class PowerOptimizer:
    def __init__(self, learning_rate=1, run=2):
        self.step = learning_rate
        self.last_gradient = None  # Initialize as None; will be updated on first call to apply_gradients
        self.not_crossed_0 = True
        self.run = run

    def apply_gradients(self, gradient):
        if self.last_gradient is None:
            self.last_gradient = np.zeros_like(gradient)

        # Initialize an empty array to store the updated steps for each gradient
        updated_gradients = np.zeros_like(gradient)

        # Iterate through each element of the gradient array
        for i, grad in enumerate(gradient):
            just_cross_0 = np.sign(self.last_gradient[i]) != np.sign(grad)

            if not just_cross_0 and self.not_crossed_0:
                self.step *= self.run
            else:
                if self.not_crossed_0:
                    self.not_crossed_0 = False
                if just_cross_0:
                    self.step /= self.run

            self.last_gradient[i] = grad
            updated_gradients[i] = self.step * np.sign(grad)

        return updated_gradients

# Testing the optimizer with an array of gradients
optimizer = PowerOptimizer()
gradient_array = np.array([0.1, -0.2, 0.3])
updated_gradients = optimizer.apply_gradients(gradient_array)
updated_gradients



