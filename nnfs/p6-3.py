import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

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


class Activation_Softmax:
    def forward(self,inputs):
        # overflow preventation

        # take all the values in output layer prior to expon.
        # substract largest value of layer from all values in layer
        # now laergest value is 0 and everything else is less than 0
        # now our range of possibliities becomes somewhere between 0 and 1 (due to nature of y=e^x)
        # actual output after exp. and norm. will be the same. ALl this does is overflow protection
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True) )  

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)
row, col = np.shape(X) # col is 2
n_neurons_1 = 3
n_neurons_2 = 3
dense1 = Layer_Dense(col, n_neurons_1)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(n_neurons_1, n_neurons_2)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output, np.sum(activation2.output, axis=1), sep="\n\n")
