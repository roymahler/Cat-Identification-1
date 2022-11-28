import matplotlib.pyplot as plt
import time
import random
from PIL import Image
import unit10.c1w2_utils as u10 #You may need to change it to: import unit10.utils as u10
import numpy as np


class perceptron(object):
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


    def sigmoid(self, arr):
        s = 1 / (1 + np.exp(-arr))
        return s


    def initialize_with_zeros(self, dim):
        W = np.zeros([dim, 1])
        b = 0
        return W, b



    def forward_propagation(self, W, b):
        #X --> array of [n, m]
        #Y --> vector with size of m for each photo in X
        #W --> vector of weights with size of n (n, 1)
        #b --> number
        m = self.X.shape[1] #number of columes of the array X (X[0])
        A = self.sigmoid(np.dot(W.T, self.X)+b) #vector of binary regression
        J = -(1/m) * (np.sum(self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A))) #cost function
        return A, J



    def backward_propagation(self, A):
        #X --> array of [n, m]
        #Y --> vector with size of m for each photo in X
        #A --> vector with the size of m of the 'activation'
        m = self.X.shape[1] #number of columes of the array X (X[0])
        dz = (1 / m) * (A - self.Y) #the sum of all the weights, x and b
        dw = np.dot(self.X, dz.T) #vector with the size of n
        db = np.sum(dz) #the dJ of the cost function (J)
        return dw, db



    def train(self, num_iterations, learning_rate):
        #X --> array of [n, m]
        #Y --> vector with size of m for each photo in X
        #num_iterations --> number of train steps
        #learning_rate --> קצב לימוד
        W, b = self.initialize_with_zeros(len(self.X))
        #alpha_W, alpha_b = np.full((len(W),1),learning_rate),learning_rate
        for i in range(num_iterations):
            A, cost = self.forward_propagation(W, b)
            dW, db = self.backward_propagation(A)
            #alpha_W = np.where(dW * alpha_W > 0, alpha_W * 1.1, alpha_W * -0.5)
            #alpha_b *= 1.1 if (db * alpha_b > 0) else -0.5
            W -= learning_rate * dW
            b -= learning_rate * db
            if i % 100 == 0:
                print("Cost after iteration {} is {}".format(i, cost))
        return W, b



    def predict(self, W, b):
        #prediction function
        Z = np.dot(W.T, self.X)+b
        return (np.where(self.sigmoid(Z) > 0.5, 1., 0.))





def CatOrNot(fname):
    W, b, num_px, classes = training()
    img = Image.open(fname)
    img = img.resize((num_px, num_px), Image.ANTIALIAS)
    plt.imshow(img)
    X = np.array(img).reshape(1, -1).T
    Y = np.array([len(X[0])])

    CatPerceptron = perceptron(X, Y)
    my_predicted_image = CatPerceptron.predict(W, b)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()


def training():
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()
    train_set_y = train_set_y.reshape(-1)
    test_set_y = test_set_y.reshape(-1)

    #setting parameters for the size of the sampled data
    m_train = train_set_y.shape[0]
    m_test = test_set_y.shape[0]
    num_px = test_set_x_orig.shape[1]

    #flatten the pictures to one dimentional array of values, keeping a seperated array for each picture
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    #normalize the values to be between 0 and 1
    train_set_x = train_set_x_flatten/255.0
    test_set_x = test_set_x_flatten/255.0

    #create my train perceptron
    Myperceptron_train = perceptron(train_set_x, train_set_y)
    Myperceptron_test = perceptron(test_set_x, test_set_y)

    #setting up W and b
    W, b = Myperceptron_train.train(num_iterations=1200, learning_rate=0.007)
    Y_prediction_train = Myperceptron_train.predict(W, b)
    Y_prediction_test = Myperceptron_test.predict(W, b)

    # Print train Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
    return W, b, num_px, classes