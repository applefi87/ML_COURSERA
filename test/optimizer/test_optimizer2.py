import numpy as np
import tensorflow as tf
# class TestOptimizer:
#     def __init__(self, learning_rate=0.1, beta=0.9,b = 1):
#         self.learning_rate = learning_rate 
#         self.beta = beta  
#         self.pgradient = float('inf')# For initial use
#         self.b = b

#     def apply_gradients(self, gradient):
#         n = self.learning_rate * gradient / ((self.beta) + (self.b-self.beta) * np.abs(gradient - self.pgradient))
#         self.pgradient = gradient
#         print("g:", gradient[0], "n:", n[0])
#         return n

class TestOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.1, beta=0.9, b=1.0, name="TestOptimizer", **kwargs):
        super(TestOptimizer, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta", beta)
        self._set_hyper("b", b)
        
        # Initialize pgradient as a large float value
        self.pgradient = tf.constant(1e10, dtype=tf.float32)

    def _create_slots(self, var_list):
        pass

    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper("learning_rate")
        beta = self._get_hyper("beta")
        b = self._get_hyper("b")
        
        n = lr * grad / (beta + (b - beta) * tf.abs(grad - self.pgradient))
        self.pgradient = grad
        
        tf.print("g:", grad[0], "n:", n[0])
        var_update = var.assign_sub(n)
        return var_update

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super(TestOptimizer, self).get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta": self._serialize_hyperparameter("beta"),
            "b": self._serialize_hyperparameter("b"),
        }


# Test the optimizer
optimizer = TestOptimizer(learning_rate=0.001)
