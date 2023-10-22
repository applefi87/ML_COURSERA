import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers

import numpy as np
units = 2
shape = 3
def generateData(length):
    xArr = []
    for _ in range(length):
        factors = np.random.uniform(-100, 101, size=shape)  # Generate 3 random numbers for each data point
        factors = factors.astype(np.float32)
        xArr.append(factors)
    return xArr
data_length = 4  # Total data length
x_data = generateData(data_length)
x_data = tf.convert_to_tensor(x_data)

# Generate corresponding y_data using random p_values
p_values = tf.random.uniform(shape=(units, shape), minval=1.0, maxval=2.0)  # Adjusted shape to (3, units)
print(x_data,p_values)
###############################################################
def is_even(tensor):
    return tf.math.mod(tensor,2) == 0

def custom_function_condensed(tensor_inputs, tensor_p):
    # sign = tf.sign(tensor_inputs)

    # p_even_mask = is_even(tensor_p)
    # print("sign",sign)
    # print("tensor_p",tensor_p)
    # print("p_even_mask",p_even_mask)
    #     #Only input is negative and p is odd then result be negative
    # result = tf.where(p_even_mask, 1.0 , sign)
    # print(result)
    sign_expanded = tf.expand_dims(tf.sign(tensor_inputs), axis=1)  
    # Broadcasting
    p_even_mask = is_even(tensor_p)
    
    # Only if input is negative and p is odd then result be negative
    # Using broadcasting to get the result for each value in tensor_p
    result = tf.where(p_even_mask | (sign_expanded == 1), 1.0, sign_expanded)
    return result

def power_fraction(inputs, p):
    ngOrPs = custom_function_condensed(inputs, p)
    
    abs_inputs_expanded = tf.expand_dims(tf.abs(inputs), axis=1)  # Shape becomes (data_length, 1, input_shape[-1])
    
    # Using broadcasting to compute the power for each value in tensor_p
    powered_values = tf.pow(abs_inputs_expanded, p)
    
    result = ngOrPs * powered_values
    return result

# print(x_data,p_values)
# y_data = tf.reduce_sum(power_fraction(x_data, p_values),2)
y_data = power_fraction(x_data, p_values)
print(y_data)
x_train = x_data
y_train = y_data

# Splitting data into training and test sets (e.g., 80% train, 20% test)
# split_index = int(0.8 * data_length)
# x_train, x_test = x_data[:split_index], x_data[split_index:]
# y_train, y_test = y_data[:split_index], y_data[split_index:]

# x_train.shape, y_train.shape, x_test.shape, y_test.shape


# x_train = generateData(3)
# x_train = tf.convert_to_tensor(x_train)

# p_values = tf.random.uniform(shape=(units,1), minval=1.0, maxval=5.0)
# y_train = power_fraction(x_train, p_values)
# print(x_train,y_train )
class PowerLayer(Layer):
    def __init__(self, units, activation='linear', **kwargs):
        super(PowerLayer, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.p = self.add_weight(shape=(self.units, input_shape[-1]),trainable=True,regularizer=regularizers.l2(0.01),name='p')
        super(PowerLayer, self).build(input_shape)
    def call(self, inputs):
        return power_fraction(inputs, self.p)
    def get_weights_params(self):
        return  self.p
    

# Build and compile the model
input_tensor = tf.keras.Input(shape=(shape,))
output_tensor = PowerLayer(units,'linear')(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.01)
optimizer =Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=MeanSquaredError())
# print("p", model.layers[1].p.numpy())
history = model.fit(x_train, y_train, epochs=4000, verbose=0)
# print( model.layers[1].p.numpy())
import time
start_time = time.time()
# Train the model
class PrintWPValueGradientAndLoss(tf.keras.callbacks.Callback):
    def __init__(self, x_train, y_train, **kwargs):
        super(PrintWPValueGradientAndLoss, self).__init__(**kwargs)
        self.x_train = x_train
        self.y_train = y_train
    def on_epoch_end(self, epoch, logs=None):
        p_value = self.model.layers[1].p.numpy()
        # Compute gradients of the loss with respect to the model's trainable variables
        with tf.GradientTape() as tape:
            predictions = self.model(self.x_train, training=True)
            loss = tf.keras.losses.mean_squared_error(self.y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Gradient for the 'w' values and 'p' values in PowerLayer
        p_gradient = grads[1]  # Index 1 corresponds to PowerLayer's 'p' weights
        # Retrieve the loss value from logs
        epoch_loss = logs["loss"]
        # Printing the information - displaying the mean values for brevity
        print(f"\nEpoch {epoch + 1} | Loss: {epoch_loss:.4f} | "
              f"Mean_p: {np.mean(p_value):.8f} | Mean_gp: {np.mean(p_gradient):.4f}")

print_p_gradient_and_loss_callback = PrintWPValueGradientAndLoss(x_train, y_train)
# history = model.fit(x_train, y_train, epochs=1000, verbose=0, callbacks=[print_p_gradient_and_loss_callback])
history = model.fit(x_train, y_train, epochs=4000, verbose=0)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds.")

# Get the value of p
print("p", model.layers[1].p.numpy())
print(x_train.numpy()[:5])
print("predict:",model.predict(x_train)[:5],"actual:",y_train.numpy()[:5])