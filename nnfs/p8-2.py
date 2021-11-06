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

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True) )  
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss: 
    def calculate(self, output, y): # output is output of model, y is intended target values
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)  # batch loss
        return data_loss 

class Loss_CategoricalCrossentropy(Loss): # inherits the loss class
    def forward(self, y_pred, y_true):
        samples = len(y_pred) # len y_true should be same anyways
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # this is for infinity prevention when we have 0
        if len(y_true.shape) == 1: # not one hot encoded values, true scalar class values
            correct_confidences = y_pred_clipped[range(samples), y_true] # explained on min 11 on : https://youtu.be/levekYbxauw?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
        elif len(y_true.shape) == 2: # one hot encoded values
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


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

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print(activation2.output, np.sum(activation2.output, axis=1), loss, sep="\n\n")
