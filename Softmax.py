import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.21, 0.2],
                 [1.41, 1.051, 0.026]]

#exponetail
exp_values = np.exp(layer_outputs)

#normalization by rows
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)) #keepdims keeps the same dimentions

print(norm_values)

'''
#no longer needed with numpy

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print (norm_values)
print (sum(norm_values))
'''

'''
Softmax - the process of exponentiation and normailization of input

ReLU doesn't work for outputs with negatives as the data is clipped and thus we loss data

Exponsital functions - solves the negative number problem by converting values exponentally without losing the meaning of the nagative value
    cons is that your data can get massive numbers adn reach overflow
    overflow pervention = u - maxu. this allows out range to be between 0 and 1
'''
