import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def is_even(tensor):
    return tf.math.mod(tensor,2) == 0

def custom_function_condensed(tensor_inputs, tensor_p):
    sign = tf.sign(tensor_inputs)
    p_even_mask = is_even(tensor_p)
    #Only input is negative and p is odd then result be negative
    result = tf.where(p_even_mask, 1.0 , sign)
    return result

def power_fraction(inputs, p):
    ngOrPs = custom_function_condensed(inputs,p)
    result = ngOrPs*tf.pow(tf.abs(inputs),p)
    return result


import numpy as np
def generateData(length):
    xArr= []
    for i in range(length):
        # factor1 = i+1
        factor1 = np.random.uniform(-100, 101)
        factor1 = np.float32(factor1)
        xArr.append([factor1])
    return xArr
x_train = generateData(8)
x_train = tf.convert_to_tensor(x_train)
y_train = power_fraction(x_train, tf.Variable(tf.fill(dims=x_train.shape, value=2.55)))
# print(x_train,y_train )

# Define custom layer
class PowerLayer(Layer):
    def __init__(self, **kwargs):
        super(PowerLayer, self).__init__(**kwargs)
        self.p = tf.Variable(initial_value=1.0, trainable=True)

    def call(self, inputs):
        return power_fraction(inputs, self.p)
    
# Build and compile the model
input_tensor = tf.keras.Input(shape=(1,))
output_tensor = PowerLayer()(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
# print( model.layers[1].p.numpy())
import time
start_time = time.time()
# Train the model
class PrintPValueGradientAndLoss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        p_value = self.model.layers[1].p.numpy()
        # Compute gradients of the loss with respect to the model's trainable variables
        with tf.GradientTape() as tape:
            predictions = self.model(x_train)
            loss = tf.keras.losses.mean_squared_error(y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Gradient for the 'p' variable in the PowerLayer
        p_gradient = grads[0].numpy()
        # Retrieve the loss value from logs
        epoch_loss = logs["loss"]
        print(f"\nEpoch {epoch + 1} | Loss: {epoch_loss:.4f} | Value of p: {p_value:.8f} | Gradient of p: {p_gradient:.4f}")

print_p_gradient_and_loss_callback = PrintPValueGradientAndLoss()
# history = model.fit(x_train, y_train, epochs=1000, verbose=0, callbacks=[print_p_gradient_and_loss_callback])
history = model.fit(x_train, y_train, epochs=4000, verbose=0)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds.")

# Get the value of p
print("p", model.layers[1].p.numpy())
print("predict:",model.predict(x_train),"actual:",y_train.numpy())