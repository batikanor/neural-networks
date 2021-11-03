import numpy as np

inputs = [1, 2, 3, 2.5]  # a vector, so mathematically denoted with 1 to 2.5 from top to bottom (vertical)

weights = [
    [0.2, 0.8, -0.5, 1.0], # for a neuron
    [0.5, -0.91, 0.26, -0.5], # for another neuron
    [-0.26, -0.27, 0.17, 0.87] # ..
]

biases = [
    2, 3, 0.5
]

print(np.shape(np.dot(weights, inputs)))

# x dot Ax = x^T times aX
# weights transpose times inputs
# so first array of weights transposed is 02, 0.5, -0.26
# and we want them to be multiplied by 1 (becuase the first weight for every neuron corresponds to the weight for the first input.) 
# simple matrix mult.
# and so on
# see nnfs_trial_1\attachments\photo_2021-11-03_21-31-21.jpg for math demonstration
# but easier way to read dot products (matr x vector) in code is to just iterate through code lines of (or arrays within) matrix and dot product them withh elements of vector (so first iteration would be 0.2 * 1 + 0.8 * 2 + . + .  and so on)
output = np.dot(weights, inputs) + biases
print(output)
