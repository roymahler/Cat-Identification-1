import matplotlib.pyplot as plt
import time
import random
from PIL import Image
import unit10.utils as u10
import numpy as np

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()

print(train_set_x_orig.shape)
train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

#print(train_set_y.shape)

# Example of a picture
index = 25 # change index to get a different picture
plt.imshow(train_set_x_orig[index])
#plt.show()
#print ("y = " + str(train_set_y[index]) + ", it's a '" + classes[np.squeeze(train_set_y[index])].decode("utf-8") +  "' picture.")

m_train = train_set_y.shape[0]
m_test = test_set_y.shape[0]
num_px = test_set_x_orig.shape[1]

'''
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
'''

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


#print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
#print ("train_set_y shape: " + str(train_set_y.shape))
#print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
#print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0


def sigmoid(arr):
    s = 1 / (1 + np.exp(-arr))
    return s

#print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim):
    W = np.zeros([dim, 1], dtype=np.float64)
    b = 0
    return W, b


#W, b = initialize_with_zeros(2)
#print ("W = " + str(W))
#print ("b = " + str(b))


def forward_propagation(X, Y, w, b):
    #X --> array of [n, m]
    #Y --> vector with size of m for each photo in X
    #w --> vector of weights with size of n (n, 1)
    #b --> number
    m = len(X[1]) #number of columes of the array X (X[0])
    A = sigmoid(np.dot(w.T, X)+b) #vector of binary regression
    J = -(1/m) * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) #cost function
    return A, J


#w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1]) 
#A, cost = forward_propagation(X, Y, w, b)
#print ("cost = " + str(cost))


def backward_propagation(X, Y, A):
    #X --> array of [n, m]
    #Y --> vector with size of m for each photo in X
    #A --> vector with the size of m of the 'activation'
    m = len(X[1]) #number of columes of the array X (X[0])
    dz = (1 / m) * (A - Y) #the sum of all the weights, x and b
    dw = np.dot(X, dz.T) #vector with the size of n
    db = np.sum(dz) #the dJ of the cost function (J)
    return dw, db


#w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]),np.array([1,0,1])
#A, cost = forward_propagation(X, Y, w, b)
#dw, db = backward_propagation(X, Y, A)
#print ("dW = " + str(dw))
#print ("db = " + str(db))


def train(X, Y, num_iterations, learning_rate):
    #X --> array of [n, m]
    #Y --> vector with size of m for each photo in X
    #num_iterations --> number of train steps
    #learning_rate --> קצב לימוד
    W, b = initialize_with_zeros(len(X))
    alpha_W, alpha_b = np.full((len(W),1),learning_rate),learning_rate
    for i in range(num_iterations):
        A, cost = forward_propagation(X, Y, W, b)
        dW, db = backward_propagation(X, Y, A)
        alpha_W = np.where(dW * alpha_W > 0, alpha_W * 1.1, alpha_W * -0.5)
        alpha_b *= 1.1 if (db * alpha_b > 0) else -0.5
        W -= alpha_W
        b -= alpha_b
        if i % 100 == 0:
            print("Cost after iteration {} is {}".format(i, cost))
    return W, b


#X, Y = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([1,0,1]) 
#W, b = train(X, Y, num_iterations= 100, learning_rate = 0.009)
#print ("W = " + str(W))
#print ("b = " + str(b))


def predict(X, w, b):
    Z = np.dot(w.T, X)+b
    return (np.where(sigmoid(Z) > 0.5, 1., 0.))

#W = np.array([[0.1124579],[0.23106775]])
#b = -0.3
#X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
#print ("predictions = " + str(predict(X, W, b)))



print(train_set_x.shape)
W, b = train(train_set_x, train_set_y, num_iterations=4000, learning_rate=0.005)
Y_prediction_test = predict(test_set_x, W, b)
Y_prediction_train = predict(train_set_x, W, b)
# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

fname = r'C:\Users\royma\source\repos\Roy_DL5\Roy_DL5\Pictures\fryd2.jpg'  # <=== change image full path
img = Image.open(fname)
img = img.resize((num_px, num_px), Image.ANTIALIAS)
plt.imshow(img)
my_image = np.array(img).reshape(1, -1).T
my_predicted_image = predict(my_image, W, b)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.show()
