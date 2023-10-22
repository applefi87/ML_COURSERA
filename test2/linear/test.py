import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
plt.style.use('./deeplearning.mplstyle')

# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# tf.autograph.set_verbosity(0)
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt

#
def generate2factorData(length):
    xArr= []
    yArr= []
    for i in range(length):
        factor1 = np.random.randint(1, 1001)
        factor2 = np.random.randint(1, 1001)
        xArr.append([factor1,factor2])
        yArr.append(factor1*factor2)
    return xArr,yArr

x_data,y_data = generate2factorData(100000)
#
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=13)
print(x_test)
#
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2)
# poly.fit(x_train)
# x_train = poly.transform(x_train)
# x_test = poly.transform(x_test)

# print(x_test)


# #
# x_train = x_train[:,[4]]
# x_test = x_test[:,[4]]
# print(x_test)
# print(y_test)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(x_test)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
################################################

print ('The first element of x_train is: ', x_train[0])

print ('The first element of y_train is: ', y_train[0])

print ('The shape of X is: ' + str(x_train.shape))
print ('The shape of y is: ' + str(y_train.shape))

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.Input(shape=(2,)),
        Dense(units=40,activation='linear',name='L1'),
        Dense(units=40,activation='linear',name='L2'),
        Dense(units=40,activation='linear',name='L4'),
        Dense(units=40,activation='linear',name='L5'),
        Dense(units=40,activation='linear',name='L6'),
        Dense(units=40,activation='linear',name='L7'),
        Dense(units=40,activation='linear',name='L8'),
        Dense(units=40,activation='linear'),
        Dense(units=40,activation='linear'),
        Dense(units=40,activation='linear'),
        Dense(units=40,activation='linear'),
        Dense(units=40,activation='linear'),
        Dense(units=40,activation='linear'),
        Dense(units=40,activation='linear'),
        Dense(units=1,activation='linear',name='L3'),
        ### END CODE HERE ### 
    ], name = "my_model" 
)

model.summary()

# [layer1, layer2, layer3, layer4] = model.layers

# W1,b1 = layer1.get_weights()
# W2,b2 = layer2.get_weights()
# W3,b3 = layer3.get_weights()
# W4,b4 = layer4.get_weights()
# print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
# print(f"W4 shape = {W4.shape}, b4 shape = {b4.shape}")

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-5),
)

history = model.fit(
    x_train,y_train,
    epochs=50
)

plot_loss_tf(history)


prediction = model.predict(x_test)  # prediction
x_exam = [[999,1000],[-999,1000]]
print(model.predict(x_exam) )
print(np.column_stack((prediction, y_test)))

print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

prediction_p = tf.nn.softmax(prediction)

print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")

yhat = np.argmax(prediction_p)

print(f"np.argmax(prediction_p): {yhat}")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell


m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
widgvis(fig)
for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1,400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()


print( f"{display_errors(model,X,y)} errors out of {len(X)} images")