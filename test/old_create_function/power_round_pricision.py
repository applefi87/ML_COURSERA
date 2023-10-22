import numpy as np

class PowerCalculator:
    def __init__(self, precision):
        self.precision = precision
    def custom_round(self, value):
        rounded_value = np.round(value, self.precision) 
        if rounded_value == np.round(rounded_value):
            return int(rounded_value) 
        else:
            return value
    def compute_power(self, x, p):
        rows, cols = x.shape
        result = np.empty_like(x, dtype=float)
        for i in range(rows):
            for j in range(cols):
                current_p = self.custom_round(p[j])
                result[i, j] = np.power(x[i,j], current_p)
        return np.squeeze(result)

# # # # Usage:
precision = 2
power_calculator = PowerCalculator(precision)

# Assume x is your input array and p is your power array
x = np.array([[1, 2], [3, 4], [5, 6]])
p = np.array([4.12,4.123])

# Call the method to compute the power
result = power_calculator.compute_power(x, p)

# Sum all the elements in the result array
result_sum = result.sum()

print(result)  # Output: the array with each element raised to the power of corresponding element in p
print(result_sum) 