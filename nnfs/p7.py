import math
# below concepts are explained on https://youtu.be/dEXPMQXoiLc?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
    # mean absolute error
    # categorical cross entrophy
        # it simplifies down to being negative log of the predicted class value as explained between min 9 and 10 in video.
    # one hot encoding
# in programming "log" usually means log base e. (ln)



softmax_output = [0.7, 0.1, 0.2]

target_output = [1, 0, 0] # so this corresponds to target class 0 (this is one hot encoding)
target_class = 0
loss = -( # categorical cross entrophy
    math.log(softmax_output[0]) * target_output[0] + 
    math.log(softmax_output[1]) * target_output[1] + # zeroes out
    math.log(softmax_output[2]) * target_output[2] # zeroes out

)
# simplifying formula above

loss = -math.log(softmax_output[target_class]) # here target_class is 0
print(loss)