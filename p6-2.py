import math 
import numpy as np

 
layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.82, 0.2],
    [1.41, 1.051, 0.026]
]


exp_values = np.exp(layer_outputs) # typically numpy functions will by default impact every value
# print(exp_values)
# print(np.sum(layer_outputs)) # in default, just gives us a scalar value here. Which is not what we want obviously
# print(np.sum(layer_outputs, axis=1, keepdims=True))
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) # sum of rows above

print(exp_values, norm_values, np.sum(norm_values, axis=1), sep="\n\n") # for softmax

