import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers

import numpy as np
data_length = 10
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

def wxpb_function(inputs, w,p,b):
    expand_dims_input =tf.expand_dims(inputs, axis=-1)
    abs_inputs_powered = tf.pow(tf.abs(expand_dims_input), p)
    isNegative = expand_dims_input < 0
    power_odd_mask = tf.math.mod(p, 2) != 0
    result_isOdd = tf.logical_and(power_odd_mask, isNegative)
    result = tf.where(result_isOdd, -abs_inputs_powered, abs_inputs_powered)
    z = w * result
    z = tf.reduce_sum(z, axis=1)+b
    # tf.matmul(w,result, transpose_a=True )
    return z

x_data = tf.random.normal(shape=(data_length, shape), dtype=tf.float32)
p_values = tf.random.uniform(shape=(shape, units), minval=3.0, maxval=3.0, dtype=tf.float32)
w_values = tf.random.uniform(shape=(shape, units), minval=2.0, maxval=2.0, dtype=tf.float32)
b_values = tf.random.uniform(shape=(units,), minval=1.0, maxval=1.0, dtype=tf.float32)

y_data = wxpb_function(x_data, w_values,p_values,b_values)

print(y_data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data.numpy(), y_data.numpy(), test_size=0.2, random_state=42)
x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

###############################################################
class PowerLayer(Layer):
    def __init__(self, units, activation='linear', **kwargs):
        super(PowerLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.p = self.add_weight(shape=[last_dim, self.units],trainable=True,regularizer=regularizers.l2(0.00000001),name='p')
        self.w = self.add_weight(shape=[last_dim, self.units],trainable=True,regularizer=regularizers.l2(0.0000001),name='w')
        self.b = self.add_weight(shape=[self.units],trainable=True,name='b')
        super(PowerLayer, self).build(input_shape)

    def call(self, inputs):
        result = wxpb_function(inputs, self.w, self.p, self.b)
        return result

# Build and compile the model
from tensorflow.keras.models import Sequential
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.Input(shape=(x_train.shape[1],)),
        PowerLayer(units=y_train.shape[1],activation='linear')
        ### END CODE HERE ### 
    ], name = "my_model" 
)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=Adam(learning_rate=0.001),
)


# input_tensor = tf.keras.Input(shape=(x_train.shape[1],))
# output_tensor = PowerLayer(y_train.shape[1],'linear')(input_tensor)
# model = Model(inputs=input_tensor, outputs=output_tensor)

# optimizer =Adam(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss=MeanSquaredError())
model.summary()
import time
start_time = time.time()
# Train the model
class PrintWPValueGradientAndLoss(tf.keras.callbacks.Callback):
    def __init__(self, x, y, **kwargs):
        super(PrintWPValueGradientAndLoss, self).__init__(**kwargs)
        self.x = x
        self.y = y
    def on_epoch_end(self, epoch, logs=None):
        p_value = self.model.layers[1].p.numpy()
        # Compute gradients of the loss with respect to the model's trainable variables
        with tf.GradientTape() as tape:
            predictions = self.model(self.x, training=True)
            loss = tf.keras.losses.mean_squared_error(self.y, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        p_gradient = grads[0] 
        epoch_loss = logs["loss"]
        print(f"\nEpoch {epoch + 1} | Loss: {epoch_loss:.15f} | "
              f"Mean_p: {np.mean(p_value):.15f} | Mean_gp: {np.mean(p_gradient):.15f}")

print_p_gradient_and_loss_callback = PrintWPValueGradientAndLoss(x_train, y_train)
# history = model.fit(x_train, y_train, epochs=4, verbose=0, callbacks=[print_p_gradient_and_loss_callback])
history = model.fit(x_train, y_train, epochs=1000, verbose=0)
# history = model.fit(x_train, y_train, epochs=4, verbose=0, callbacks=[print_p_gradient_and_loss_callback])
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds.")

# Get the value of p
p_actual = p_values.numpy()
p_predicted = model.layers[0].p.numpy()


def compare_values(p_actual, p_predicted):
    # Print headers
    print(f"{'p_actual':<20} {'p_predicted':<20} {'Difference':<20}")
    print('-'*60)
    max_diff = 0.0
    for i in range(p_actual.shape[0]):
        for j in range(p_actual.shape[1]):
            actual = p_actual[i][j]
            predicted = p_predicted[i][j]
            diff = abs(actual - predicted)
            # Update max difference if current difference is greater
            if diff > max_diff:
                max_diff = diff
            # print(f"{actual:<20.8f} {predicted:<20.8f} {diff:<20.8f}")
    print('\nMaximum Difference:', max_diff)
compare_values(p_values.numpy(), model.layers[0].p.numpy())



print("p_actual", p_values.numpy())
print("p", model.layers[0].p.numpy())
print("x",x_train[:5])
print("predict:",model.predict(x_test)[:5],"actual:",y_test[:5])