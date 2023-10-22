import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers

###############################################################
class DenseSquareLayer(Layer):
    def __init__(self, units, activation='linear',pow=1, **kwargs):
        super(DenseSquareLayer, self).__init__(**kwargs)
        self.units = units
        self.pow = pow
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.w = self.add_weight(shape=[last_dim, self.units],trainable=True,regularizer=regularizers.l2(0.00001),name='w')
        self.b = self.add_weight(shape=[self.units],trainable=True,name='b')
        super(DenseSquareLayer, self).build(input_shape)
    def call(self, inputs):
        result = tf.matmul(tf.pow(inputs,self.pow), self.w) +self.b
        return result