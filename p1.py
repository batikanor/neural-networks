# assume three neurons are feeding to this neural network that we build


# inputs = outputs from the previous layer
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

# output of this layer
output = inputs[0] * weights[0] + \
inputs[1] * weights[1] + \
inputs[2] * weights[2] + bias

print(output)