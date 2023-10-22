import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers

import numpy as np
data_length = 1
shape = 2
units = 3
def generateData(length):
    xArr = []
    for _ in range(length):
        factors = np.random.uniform(-100, 101, size=shape)  # Generate 3 random numbers for each data point
        factors = factors.astype(np.float32)
        xArr.append(factors)
    return xArr
x_data = generateData(data_length)
x_data = tf.convert_to_tensor(x_data)
# print(x_data)

## 哪個對?
def wxpb_function_original(inputs, w,p,b):
    expand_dims_input =tf.expand_dims(inputs, axis=-1)
    abs_inputs_powered = tf.pow(tf.abs(expand_dims_input), p)
    isNegative = expand_dims_input < 0
    power_odd_mask = tf.math.mod(p, 2) != 0
    result_isOdd = tf.logical_and(power_odd_mask, isNegative)
    result = tf.where(result_isOdd, -abs_inputs_powered, abs_inputs_powered)
    z = w * result
    z = tf.reduce_sum(z, axis=1)
    return z


def wxpb_function_optimized(inputs, w,p,b):
    expand_dims_input =tf.expand_dims(inputs, axis=-1)
    abs_inputs_powered = tf.pow(tf.abs(expand_dims_input), p)
    isNegative = expand_dims_input < 0
    power_odd_mask = tf.math.mod(p, 2) != 0
    result_isOdd = tf.logical_and(power_odd_mask, isNegative)
    result = tf.where(result_isOdd, -abs_inputs_powered, abs_inputs_powered)
    z = tf.matmul(w,result, transpose_a=True )
    z = tf.reduce_sum(z,axis=1)
    return z

x_data = tf.random.normal(shape=(data_length, shape), dtype=tf.float32)
p_values = tf.random.uniform(shape=(shape, units), minval=1.0, maxval=5.0, dtype=tf.float32)
w_values = tf.random.uniform(shape=(shape, units), minval=1.0, maxval=5.0, dtype=tf.float32)
b_values = tf.random.uniform(shape=(units,), minval=1.0, maxval=5.0, dtype=tf.float32)
print(p_values,w_values,b_values)
print(x_data)

y_data = wxpb_function_original(x_data, w_values,p_values,b_values)
y2_data = wxpb_function_optimized(x_data, w_values,p_values,b_values)
 
print(y_data,y2_data)

import timeit
def time_optimized():
    wxpb_function_optimized(x_data,w_values, p_values,b_values)

time_optimized_execution = timeit.timeit(time_optimized, number=2000)

# Time the execution of the original function
def time_original():
    wxpb_function_original(x_data,w_values, p_values,b_values)

time_original_execution = timeit.timeit(time_original, number=2000)
# Time the execution of the optimized function





print(f"Original function execution time for 1000 runs: {time_original_execution:.6f} seconds")
print(f"Optimized function execution time for 1000 runs: {time_optimized_execution:.6f} seconds")



output_original = wxpb_function_original(x_data,w_values, p_values,b_values)
output_optimized = wxpb_function_optimized(x_data,w_values, p_values,b_values)

# Check if the results are close to each other
tolerance = 1e-5
are_results_close = tf.reduce_all(tf.abs(output_original - output_optimized) < tolerance)

# Convert the differences tensor to numpy for easier manipulation and printing
# Compute the absolute differences between the outputs of the two functions
differences = tf.abs(output_original - output_optimized)

# Convert the differences tensor to numpy for easier manipulation and printing
differences_np = differences.numpy()

# Print the differences for each data point
print("Absolute Differences for Each Data Point:")
print('-' * 40)
for i, row_diff in enumerate(differences_np):
    for j, diff in enumerate(row_diff):
        print(f"Data Point ({i + 1}, {j + 1}): {diff:.8f}")

# You can also print some statistics about the differences
print('\nStatistics:')
print('-' * 40)
print(f"Maximum Difference: {np.max(differences_np):.8f}")
print(f"Minimum Difference: {np.min(differences_np):.8f}")
print(f"Mean Difference: {np.mean(differences_np):.8f}")
print(f"Standard Deviation of Differences: {np.std(differences_np):.8f}")
