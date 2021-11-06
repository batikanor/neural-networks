import numpy as np

softmax_outputs = np.array([
     [0.7, 0.1, 0.2],
     [0.1, 0.5, 0.4],
     [0.02, 0.9, 0.08]
])

class_targets = [0, 1, 1] # from 0.7, 0.5 and 0.9 above

# print(softmax_outputs[[0,1,2], class_targets])
print(
    softmax_outputs[
        range(len(softmax_outputs)), class_targets
    ]
)
print()
neg_log = -np.log(
        softmax_outputs[
            range(len(softmax_outputs)), class_targets
        ]
    )

# loss of this batch -> mean of the losses
print()

average_loss = np.mean(neg_log)
print(neg_log, average_loss)

# issue: negative log of 0 is infinite
    # if confidence of coorrect class is 0, then it is already a funny thing.
    # the problem is that if there is an inf (infinity) value in numpy .mean, it will just be infiniity
        # solution: clip the values in the range by some insignificant amount
