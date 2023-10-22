import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers

def wxpb_function(inputs, w, p, b, epsilon=1e-9):
    expand_dims_input = tf.expand_dims(inputs+ epsilon, axis=-1)
    # tf.print("expand_dims_input:", expand_dims_input)
    abs_inputs_powered = tf.pow(tf.abs(expand_dims_input), p)
    # tf.print("abs_inputs_powered:", abs_inputs_powered)
    isNegative = expand_dims_input < 0
    power_odd_mask = tf.math.mod(p, 2) != 0
    result_isOdd = tf.logical_and(power_odd_mask, isNegative)
    result = tf.where(result_isOdd, -abs_inputs_powered, abs_inputs_powered)
    # tf.print("result:", result)
    z = w * result
    z = tf.reduce_sum(z, axis=1) + b
    return z

from tensorflow.keras import activations
class PowerLayer(Layer):
    def __init__(self, units, activation='linear',kernel_regularizer=None, **kwargs):
        super(PowerLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        initializer = tf.keras.initializers.Constant(1.0)
        self.p = self.add_weight(shape=[last_dim, self.units],trainable=True,initializer=initializer,regularizer=self.kernel_regularizer,name='p')
        self.w = self.add_weight(shape=[last_dim, self.units],trainable=True,regularizer=self.kernel_regularizer,name='w')
        self.b = self.add_weight(shape=[self.units],trainable=True,name='b')
        super(PowerLayer, self).build(input_shape)

    def call(self, inputs):
        result = wxpb_function(inputs, self.w, self.p, self.b)
        result = self.activation(result)
        return result
