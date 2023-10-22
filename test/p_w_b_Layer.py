import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers

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
x_train = generateData(30)
x_train = tf.convert_to_tensor(x_train)
y_train = power_fraction(x_train, tf.Variable(tf.fill(dims=x_train.shape, value=2.55)))*2.55
# print(x_train,y_train )

def safe_power_fraction(inputs, p):
    # Clip the values of p to a range to prevent extreme values
    # p_clipped = tf.clip_by_value(p, -4.0, 4.0)  # Clipping the powers to a range
    ngOrPs = custom_function_condensed(inputs, p)
    result = ngOrPs * tf.pow(tf.abs(inputs), p)
    # Replace any inf or NaN values with a large number
    # result = tf.where(tf.math.is_finite(result), result, 9e10)
    # print(result)
    return result

class PowerLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='linear', **kwargs):
        super(PowerLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        # Weight values
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 trainable=True,
                                 regularizer=regularizers.l2(0.01),
                                 name='w')
        # Power values
        self.p = self.add_weight(shape=(input_shape[-1], self.units),
                                 trainable=True,
                                 regularizer=regularizers.l2(0.01),
                                 name='p')
        super(PowerLayer, self).build(input_shape)
    def call(self, inputs):
        x_powered = safe_power_fraction(inputs, self.p)
        x = self.w * x_powered
        return self.activation(x)
    def get_weights_params(self):
        return self.w, self.p
    
# Build and compile the model
input_tensor = tf.keras.Input(shape=(1,))
output_tensor = PowerLayer(1,'linear')(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.01)
optimizer =Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss=MeanSquaredError())
print("p", model.layers[1].p.numpy())
print("w", model.layers[1].w.numpy())

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
        # Retrieve w_values and p_values from the PowerLayer
        w_value = self.model.layers[1].w.numpy()
        p_value = self.model.layers[1].p.numpy()
        
        # Compute gradients of the loss with respect to the model's trainable variables
        with tf.GradientTape() as tape:
            predictions = self.model(self.x_train, training=True)
            loss = tf.keras.losses.mean_squared_error(self.y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        # Gradient for the 'w' values and 'p' values in PowerLayer
        w_gradient = grads[0]  # Index 0 corresponds to PowerLayer's 'w' weights
        p_gradient = grads[1]  # Index 1 corresponds to PowerLayer's 'p' weights
        
        # Retrieve the loss value from logs
        epoch_loss = logs["loss"]
        
        # Printing the information - displaying the mean values for brevity
        print(f"\nEpoch {epoch + 1} | Loss: {epoch_loss:.4f} | "
              f"Mean_w: {np.mean(w_value):.8f} | Mean_gw: {np.mean(w_gradient):.4f} | "
              f"Mean_p: {np.mean(p_value):.8f} | Mean_gp: {np.mean(p_gradient):.4f}")

print_p_gradient_and_loss_callback = PrintWPValueGradientAndLoss(x_train, y_train)
# history = model.fit(x_train, y_train, epochs=1000, verbose=0, callbacks=[print_p_gradient_and_loss_callback])
history = model.fit(x_train, y_train, epochs=4000, verbose=0)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds.")

# Get the value of p
print("p", model.layers[1].p.numpy())
print("w", model.layers[1].w.numpy())
print(x_train.numpy()[:5])
print("predict:",model.predict(x_train)[:5],"actual:",y_train.numpy()[:5])