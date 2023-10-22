import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers

import numpy as np
data_length = 2  # Total data length
units = 1
shape = 1
def generateData(length):
    xArr = []
    for _ in range(length):
        factors = np.random.uniform(-100, 101, size=shape)  # Generate 3 random numbers for each data point
        factors = factors.astype(np.float32)
        xArr.append(factors)
    return xArr
x_data = generateData(data_length)
x_data =[[0.0],[-1.0]]
x_data = tf.convert_to_tensor(x_data)
print(x_data)


def power_fraction(inputs, powers):
    isNegative = (inputs)[:, :, None] <0
    power_odd_mask = tf.math.mod(powers, 2) != 0
    result_isOdd = tf.logical_and(power_odd_mask ,isNegative)
    abs_inputs_powered = tf.pow(tf.abs(inputs)[:, :, None], powers)
    result = tf.where(result_isOdd , -abs_inputs_powered, abs_inputs_powered)
    return result 


# Parameters
x_data = tf.random.normal(shape=(data_length, shape), dtype=tf.float32)
p_values = tf.random.uniform(shape=(shape, units), minval=2.0, maxval=2.0, dtype=tf.float32)

y_data = power_fraction(x_data, p_values)
print("x:",x_data)
print("p:",p_values)
print("y:",y_data)