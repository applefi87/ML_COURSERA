import numpy as np
#
def generate2factorData(length):
    xArr= []
    yArr= []
    for i in range(length):
        factor1 = i+1
        factor2 = np.random.randint(1, 1001)
        xArr.append([factor1,factor2])
        yArr.append(factor1*factor2)
    return xArr,yArr

#
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
  def compute_loss(self,y_pred,y,p,_lambda):
    # print(y_pred.shape)
    return ((y_pred-y)**2).mean()/2 + _lambda*np.sum(p**2)/(2*len(y))
  
  def gradient(self,x, y_pred, y,p,_lambda):
    x_no0 = x
    x_no0 = np.where(x < 1e-5, 1e-5, x)  # replace zeros prevent 0**-2 or log(0)
    x_power_p = np.power(x_no0,p)
    log_x = np.log(x_no0)
    # print("x_power_p:",x_power_p)
    # print("log_x:",log_x)
    # print("yd:",y_pred-y)
    dp = np.dot((x_power_p*log_x).T,y_pred-y)/len(y)+ _lambda*p/len(y)
    # print("dP:",np.dot((x_power_p*log_x).T,y_pred-y).shape)
    # print(dp.shape)
    return dp

class GradientDescentOptimizer:
  def __init__(self,learning_rate,_lambda):
    self.learning_rate=learning_rate
    self._lambda=_lambda
  def update_params(self,x,y,model,cost_fun,iter):
    for i in range(iter+1):
      y_pred = model.predict(x)
      p = model.get_params()
      # print("p:",p)
      cost = cost_fun.compute_loss(y_pred,y,p,self._lambda)
      dp = cost_fun.gradient(x,y_pred,y,p,self._lambda)
      p= p- dp*self.learning_rate
      model.update_params(p)
      if(i%np.floor(iter/8)==0):
        print(f"Iter:{i: 5}  cost:{cost: .2f}  p:{p} ")
        print("-----")
    return 
#
class LinearModel:
  def __init__(self,x_data):
    self.p = self.p = np.full(len(x_data[0]), 2)
  def predict(self,x):
    return np.power(x,self.p).sum()
  def update_params(self,new_power):
        self.p = new_power
  def get_params(self):
    return self.p
x_train,y_train = generate2factorData(1)
#
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2,include_bias=False)
poly.fit(x_train)
x_train = poly.transform(x_train)
# print(x_test)
#
x_train = x_train[:,[3]]
print(x_train)
print(y_train)
x_train = np.array(x_train)
y_train = np.array(y_train)

model = LinearModel(x_train)

gd = GradientDescentOptimizer(1e-9,0)
# gd = GradientDescentOptimizer(1e-8,0)
gd.update_params(x_train,y_train,model,MSE(),80000)

p = model.get_params()
print(p)


# tw = np.random.randn(2)
# tp = np.random.randn(2)

# print(np.dot(tw,tp))
# print(tw*tp)
# print(np.log(tw))