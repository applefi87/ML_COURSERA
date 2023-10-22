# Redefining the AdamOptimizer class

import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Initialize first and second moment vectors
        self.m = 0
        self.v = 0
        # Initialize timestep
        self.t = 0
    
    def apply_gradients(self, gradient):
        # Increment the timestep
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        
        # Compute bias-corrected first moment estimate
        m_corrected = self.m / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second raw moment estimate
        v_corrected = self.v / (1 - self.beta2 ** self.t)
        
        # Compute the parameter update
        dp = -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        return dp

# Now, running the optimization test again
# x = 5.0
# optimizer = AdamOptimizer(learning_rate=0.1)
# x_values = [x]
# f_values = [f(x)]
# for _ in range(num_iterations):
#     gradient = gradient_f(x)
#     dx = optimizer.apply_gradients(gradient)
#     x += dx
#     x_values.append(x)
#     f_values.append(f(x))

# x_values[-10:], f_values[-10:]  # Displaying the last 10 values of x and f(x)