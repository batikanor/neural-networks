import math 
layer_outputs = [4.8, 1.21, 2.385]
# layer_outputs = [4.8, 4.79, 4.25]

# softmax activation function is explained on minute 8 of video https://youtu.be/omz_NdFgWyU?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
    # in one layer (specifically output layer) we may need a probability distribution
    # y = e^x gives us always positive numbers and therefore solves our negativity problem with linear activation functions.
    # and the problem with ReLU here is that it clips some data off and some info is therefore lost etc etc


E = math.e 
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)
# print(exp_values)

# first we've exponentiated and then we'll normalize. Thats because we don't want to lose the meaning of any (neg.) value

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)
print(norm_values, sum(norm_values))

