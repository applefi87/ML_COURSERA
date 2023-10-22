import numpy as np
# np.set_printoptions(precision=7)

#
import sympy as sp
# w,p,x,n =sp.symbols("w p x n")
# f = (w*np.power(x,p)+n)**2
# df_dw = sp.diff(f,w)
# df_dp = sp.diff(f,p)
# print(df_dw,df_dp)

# y,p,x=sp.symbols("y p x")
# f =  ((x**p-y)**2)
# df_dp = sp.diff(f,p)
# print(df_dp)

class MSE:
    """
    Mean Squared Error (MSE) class with regularization for computing loss and gradient.
    """
    def compute_loss(self, y_pred, y, p, _lambda):
        """
        :param p: Model parameters.
        """
        squared_error = ((y_pred - y) ** 2).mean()
        regularization_term = _lambda * np.sum(p ** 2) 
        return  (squared_error + regularization_term)/(2*len(y))

    def gradient(self, x, y_pred, y, p, _lambda):
        """
        :param p: Model parameters.
        """
        # Replace values close to zero to avoid division by zero in log
        # x_no0 = np.where(x < 1e-5, 1e-5, x)
        x_no0 =x
        # Compute gradient components
        x_power_p = np.power(x_no0, p)
        log_x = np.log(x_no0)
        error_gradient = np.dot((x_power_p * log_x).T, y_pred - y) / len(y)
        regularization_gradient = _lambda * p / len(y)
        
        return error_gradient + regularization_gradient

class GradientBooster:
    """Gradient Booster class to smooth and boost the gradient updates."""
    def __init__(self, power_precision=3, new_gradient_rate=1.4):
        """ 
        :param power_precision: Smaller values lead to faster descent. Should be a positive integer.
        :param new_gradient_rate: Scaling factor for the gradient update. Larger values lead to bigger steps allowed, but too large might lead to imprecise steps.
        """
        self.beta = 1 - (10 ** -power_precision)
        self.new_gradient_beta = 1 - self.beta
        self.ema_gradient = 0
        self.new_gradient_rate = new_gradient_rate
    def boost_gradient(self, gradient):
        self.ema_gradient = self.beta * self.ema_gradient + self.new_gradient_beta  * gradient * self.new_gradient_rate
        return self.ema_gradient
        """
        # Can use here to test
        # num = 0
        # power_precision =3
        # beta = 1 - (10 ** -power_precision)
        # imaginary_gradient = 1
        # new_gradient_rate=2
        # for i in range(10000):
        #   num = num*beta+(1-beta)*imaginary_gradient*new_gradient_rate
        #   print(num)
        # print(num)  
        """
import tensorflow as tf
from scipy.special import expit
# from collections import deque
# dp_deque =deque(maxlen=2)
class GradientDescentOptimizer:
    def __init__(self, power_precision, learning_rate, _lambda):
        """
        :param power_precision: Power precision value for the Result power.
        """
        from Adam_optimizer import AdamOptimizer 
        self.gb = GradientBooster(power_precision-1)
        self.power_precision = power_precision
        self.power_precision_ten = 10 ** power_precision
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.optimizer = AdamOptimizer(0.6)
    def update_params(self, x, y, model, cost_fun, iter):
        for i in range(iter + 1):
            y_pred, p = self._forward_pass(x, model)
            cost, dp = self._compute_cost_and_gradient(x, y_pred, y, p, cost_fun)
            fix_dp, boosted_gradient = self._calculate_fixed_gradient(dp)
            new_dp = self.optimizer.apply_gradients(dp)
            model.update_params(new_dp)
            if i % np.ceil(iter / 8) == 0:
                self._print_progress(i, cost, p, fix_dp, boosted_gradient)
    def _forward_pass(self, x, model):
        """
        Perform forward pass.

        :param x: Input data.
        :param model: Model to be used.
        :return: Predicted values and model parameters.
        """
        y_pred = model.predict(x)
        p = model.get_params()
        return y_pred, p

    def _compute_cost_and_gradient(self, x, y_pred, y, p, cost_fun):
        """
        Compute cost and gradient.
        :param p: Model power parameters.
        """
        cost = cost_fun.compute_loss(y_pred, y, p, self._lambda)
        dp = cost_fun.gradient(x, y_pred, y, p, self._lambda)
        return cost, dp
      
    def _calculate_fixed_gradient(self, dp):
        sigmoid = expit(dp)
        fix_dp = (sigmoid - 0.5) * 2 / self.power_precision_ten
        boosted_gradient = self.gb.boost_gradient(fix_dp)
        return fix_dp, boosted_gradient

    def _print_progress(self, iteration, cost, p, fix_dp, boosted_gradient):
        print(f"-Iter:{iteration: 5}  cost:{cost: .2f}  p:{p} ")
        # print("fix_dp:", fix_dp,"boost_gradient(fix_dp)", boosted_gradient)
#


class LinearModel:
    """
    A linear model that uses powers of the input data for predictions.
    
    Attributes:
        p (numpy array): Model parameters.
        power_precision (int): Precision for power calculations.
        rp (object): An instance of a class responsible for rounded power calculations (assumed).
    """
    def __init__(self, x_data, power_precision):
        """
        :param x_data: Input data for create params shape.
        :param power_precision: Precision for power calculations.
        """
        from power_round import PowerCalculator as rounded_power
        self.p = np.full(len(x_data[0]), 1)
        self.power_precision = power_precision
        self.rp = rounded_power(power_precision)

    def predict(self, x):
        return np.squeeze(np.power(x, self.p))

    def build_predict(self, x):
        """
        Build the prediction using rounded power calculations.
        Prevent float power as np.power(4,4.) have large error
        """
        return self.rp.compute_power(x, self.p)
    def update_params(self, dp):
        self.p =  self.p+ dp
    def get_params(self):
        return np.round(self.p,self.power_precision)

def generateData(length):
    xArr= []
    yArr= []
    for i in range(length):
        # factor1 = i+1
        factor1 = np.random.randint(1, 101)
        xArr.append([factor1])
        yArr.append(factor1**5.0065)
    return xArr,yArr

#
import time

x_train,y_train = generateData(10)
x_train = np.array(x_train)
y_train = np.array(y_train)
# print(x_train)
# print(y_train)
start_time = time.time()
power_precision = 10
model = LinearModel(x_train,power_precision)
gd = GradientDescentOptimizer(power_precision,1e-5,0)
gd.update_params(x_train,y_train,model,MSE(),128)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds.")


print(1-2**-60)
# p = p.astype(np.float64)
p = model.get_params()
print("p",p)
y_pred2 = model.build_predict(x_train)
print(x_train)
print(y_train)
print(y_pred2)



