import numpy as np

np.random.seed(0)

# X is for inputs (input feature sets are conventionally denoted with X)
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# when you save a model you're saving the weights and biases
# when you load, you simply load them
# but for initialization, you initialize them with rnadom values but from a certrain range, mostly small values etc
# keep values small rather than letting them explode.
# for values, initialize them with values in range -0.1 and 0.1
# for biases just initialize them with 0 (for some cases you may want to have biases that are non-zero, e.g. when your network is like dead and all outputs are 0 etc etc)
class Layer_Dense: # dense layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # on p3.py the shape of weights was number of neurons by the number of weights, but here it'll be the opposite (so the transpose of it right away)
        self.biases = np.zeros((1, n_neurons)) # we pass the shape as a parameter (the tuple (1,n_neurons))
    def forward(self, inputs):
        # inputs could be first input data or output of previous layer
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

# layer1 = Layer_Dense(4,5)
row, col = np.shape(X)
n_neurons_1 = 5
n_neurons_2 = 2
layer1 = Layer_Dense(col, n_neurons_1)

layer2 = Layer_Dense(n_neurons_1, n_neurons_2) 

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)