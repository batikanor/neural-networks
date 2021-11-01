# now we're modeling 3 neurons with 4 inputs
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5




# output of this layer
output = [
    inputs[0] * weights1[0] + \
    inputs[1] * weights1[1] + \
    inputs[2] * weights1[2] + \
    inputs[3] * weights1[3] + bias1, 
    inputs[0] * weights2[0] + \
    inputs[1] * weights2[1] + \
    inputs[2] * weights2[2] + \
    inputs[3] * weights2[3] + bias2, 
    inputs[0] * weights3[0] + \
    inputs[1] * weights3[1] + \
    inputs[2] * weights3[2] + \
    inputs[3] * weights3[3] + bias3
]

print(output)



# assume you're trying to predict failure or not failure for computer equipment
# each unique value in inputs could be value from a separrate sensor in that exapmle
# your inputs from a neuron could either be such values or they could be outputs from neurons themselves