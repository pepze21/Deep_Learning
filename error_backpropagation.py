# import warnings
# warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt


# 소숫점 3자리까지만 & 과학적표기법 억제
np.set_printoptions(suppress=True, precision=3) 

X = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])

y = np.array([0., 1., 1., 0.]).reshape(4, 1)

np.random.seed(2045) # seed == 2045
W1 = np.random.rand(3, 4)

np.random.seed(2046) # seed == 2046
W2 = np.random.rand(4, 1)

np.random.seed(2045)
y_hat = np.random.rand(4).reshape(4, 1)

Layer1 = np.ones([4, 4]) # default == float64

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1.0 - x)

# MSE
# def loss_function(y, y_hat):
#     return np.mean((y - y_hat) ** 2)

# the mean of Binary Cross Entropy Error
def loss_function(y, y_hat):
    return -np.mean((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

# Forward Propagation
def forwardProp(X, W1, Layer1, W2, y_hat):
    Layer1 = sigmoid(np.dot(X, W1))
    y_hat = sigmoid(np.dot(Layer1, W2))
    return Layer1, y_hat

# Error Back Propagation
def backProp(X, y, y_hat, Layer1, W1, W2):
    d_W2 = np.dot(np.transpose(Layer1), (-2 * (y - y_hat) * d_sigmoid(y_hat)))
    d_W1 = np.dot((-2 * (y - y_hat) * d_sigmoid(y_hat)), np.transpose(W2))
    d_W1 = d_W1 * d_sigmoid(Layer1)
    d_W1 = np.dot(np.transpose(X), d_W1)
    W1 = W1 - 0.8 * d_W1
    W2 = W2 - 0.8 * d_W2
    return y_hat, Layer1, W1, W2

loss_record = []

# learning
for k in range(0, 1000):
    Layer1, y_hat = forwardProp(X, W1, Layer1, W2, y_hat)
    y_hat, Layer1, W1, W2 = backProp(X, y, y_hat, Layer1, W1, W2)
    loss_record.append(loss_function(y, y_hat))

plt.figure(figsize=(9, 6))
plt.plot(loss_record)
plt.show()
