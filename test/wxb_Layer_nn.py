import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers

import numpy as np
data_length = 500
shape = 100
units = 40
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

x_data = tf.random.normal(shape=(data_length, shape), dtype=tf.float32)
w_values = tf.random.uniform(shape=(shape, units), minval=1.0, maxval=5.0, dtype=tf.float32)
b_values = tf.random.uniform(shape=(units,), minval=1.0, maxval=5.0, dtype=tf.float32)

y_data = tf.matmul(x_data, w_values)+b_values
# print(x_data ,y_data )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data.numpy(), y_data.numpy(), test_size=0.2, random_state=42)
x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

###############################################################
class DenseLikeLayer(Layer):
    def __init__(self, units, activation='linear', **kwargs):
        super(DenseLikeLayer, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        initializer = tf.keras.initializers.Constant(1.0)
        self.w = self.add_weight(shape=[last_dim, self.units],initializer=initializer,trainable=True,regularizer=regularizers.l2(0.00001),name='w')
        self.b = self.add_weight(shape=[self.units],initializer=initializer,trainable=True,name='b')
        super(DenseLikeLayer, self).build(input_shape)
    def call(self, inputs):
        result = tf.matmul(inputs, self.w) +self.b
        return result
    
# Build and compile the model
input_tensor = tf.keras.Input(shape=(shape,))
output_tensor = DenseLikeLayer(units,'linear')(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)

optimizer =Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=MeanSquaredError())
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
        w_value = self.model.layers[1].w.numpy()
        # Compute gradients of the loss with respect to the model's trainable variables
        with tf.GradientTape() as tape:
            predictions = self.model(self.x, training=True)
            loss = tf.keras.losses.mean_squared_error(self.y, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        w_gradient = grads[0] 
        epoch_loss = logs["loss"]
        print(f"\nEpoch {epoch + 1} | Loss: {epoch_loss:.15f} | "
              f"Mean_w: {np.mean(w_value):.15f} | Mean_gw: {np.mean(w_gradient):.15f}")

print_p_gradient_and_loss_callback = PrintWPValueGradientAndLoss(x_train, y_train)
# history = model.fit(xtrain, y_train, epochs=4, verbose=0, callbacks=[print_p_gradient_and_loss_callback])
history = model.fit(x_train, y_train, epochs=500, verbose=0)
# history = model.fit(x_train, y_train, epochs=4, verbose=0, callbacks=[print_p_gradient_and_loss_callback])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds.")

# Get the value of p
w_actual = w_values.numpy()
w_predicted = model.layers[1].w.numpy()

def compare_values(actual_vals, predicted_vals):
    """
    Display the actual and predicted values side by side for easy comparison.
    
    Parameters:
    - actual_vals: numpy array containing the actual values
    - predicted_vals: numpy array containing the predicted values
    """
    
    # Print headers
    # print(f"{'actual':<20} {'predicted':<20} {'Difference':<20}")
    print('-'*60)

    max_diff = 0.0

    for i in range(actual_vals.shape[0]):
        for j in range(actual_vals.shape[1]):
            actual = actual_vals[i][j]
            predicted = predicted_vals[i][j]
            diff = abs(actual - predicted)

            # Update max difference if current difference is greater
            if diff > max_diff:
                max_diff = diff

            # Uncomment the next line if you want to print each value
            # print(f"{actual:<20.8f} {predicted:<20.8f} {diff:<20.8f}")

    print('\nMaximum Difference:', max_diff)
compare_values(w_values.numpy(), model.layers[1].w.numpy())
# compare_values(b_values.numpy(), model.layers[1].b.numpy())
print(w_values.numpy())

# print("p_actual", p_values.numpy())
# print("p", model.layers[1].p.numpy())
print("x",x_train[:5])
print("predict:",model.predict(x_test)[:5],"actual:",y_test[:5])