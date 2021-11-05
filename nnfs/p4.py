# Until now we've only modeled a single layer of neuron, so lets go for 2 layers now, and think about how we may extrapolate out to supporting an arbitrary number of layers

import numpy as np

# gpus are used more often for machine learning because they have more cores than cpus. And that allows us to run calculations on parallel (in batches etc)


# batches help with generalization also

# inputs = [1,2,3,2.5] # these could be reading from sensors etc
# inputs above are a single sample, but we should be passing a batch of these data instead for most use cases.
# we should show the algorithm more samples at once so that it generalizes better
# see (05:40) https://youtu.be/TEWy9vZcxW4?list=RDCMUCfzlCWGWYyIQ0aLC5w48gBQ to see the effect of raising sample size.
# if you show all the samples at once that could cause overfitting. (hurts generalization). It will probably perform bad for out-of-sample data.
# batch size of 32 is a pretty common batch size but it depends on the context.

inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
] # now both the inputs and weights will be matrices. so it is matrix .(dot product) matrix


weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
] # we need to transpose weights now. np.dot already converts them to numpy arrays but here to make it easy to transpose we'll do it right away
# explanation about why we need transpose here (14:50 https://youtu.be/TEWy9vZcxW4?list=RDCMUCfzlCWGWYyIQ0aLC5w48gBQ)
biases = [2, 3, 0.5]


output = np.dot(inputs, np.array(weights).T) + biases
print(output)
