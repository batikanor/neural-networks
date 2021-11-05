# expalanation about activation functions in the first 2 minutes: https://youtu.be/gmjzbpSVY1A?list=RDCMUCfzlCWGWYyIQ0aLC5w48gBQ


# following activation functions come in after you do the inputs*weights + bias and are explained on video linked above

# step activation function
# sigmoid activation function
# rectified linear activation function(unit) (ReLU) -> this is granular. (sigmoid is too) sigmoid has an issue called vanishing gradient problem and this may help with it
    # relu is a simple calculation and faster than sigmoid
    # it just simply works
# just using weights and biases -> it is like our activation function is a linear one with y=x... All outputs will be linear. We'll only be able to fit linear function and if we try to fit linear data all we can do is approximate using some linear function.
    # example shown at 7:38 on video linked on top comment. linear activation function cannot really approxiate sin wave well.
        # how relu performs better can be seen on 8:20
            # why it works is that ReLU is non linear. It is almost linear but yet that little bit of rectifiedness (clipping at zero) is exactly what makes it maybe almost as powerful as sigmoid activation function

# relu is demonstrated well on video linked to on top

import numpy as np
from nnfs.datasets import spiral_data
import nnfs   
nnfs.init() # what this does is the following
#     #1. it sets up np.random.seed(0)
#     #2. dot product sometimes uses a different data type and there is no way to set a default data type in numpy
#         #2.1. so this overrides somethings to nsure that data types are the same as in the book (nnfs)
#     #3. The dataset generationgasdg
# # inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2 -100]
# # output = []
# # # relu
# # for i in inputs:
# #     output.append(max(0,i))
# # print(output)

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# This function generates data for us, it just makes things easy for learning. it is also a part of the package nnfs
# function below is a modified version of code in cs231 course
def create_data(points, classes):
    X  = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix  = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y 


class Layer_Dense: # dense layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # on p3.py the shape of weights was number of neurons by the number of weights, but here it'll be the opposite (so the transpose of it right away)
        self.biases = np.zeros((1, n_neurons)) # we pass the shape as a parameter (the tuple (1,n_neurons))
    def forward(self, inputs):
        # inputs could be first input data or output of previous layer
        self.output = np.dot(inputs, self.weights) + self.biases
        pass


class Activation_ReLU:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



X, y = create_data(100, 3)
X, y = spiral_data(100, 3) # 100 feature sets, 3 classes
row, col = np.shape(X) # col is 2
n_neurons_1 = 5
n_neurons_2 = 2
print(col, n_neurons_1)
layer1 = Layer_Dense(col, n_neurons_1)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(n_neurons_1, n_neurons_2) 


layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
layer2.forward(activation1.output)
# print(layer2.output)