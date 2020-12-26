import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
#3 neurons
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

#dot multiplies weights and inputs together and add up the sums
output = np.dot(weights, inputs) + biases
#does dot product 3 times (weights[n] * inputs)
#then takes that and add bias values
print(output)

'''
notes:
weights go first because thats the first to be "indexed", we want weights to be indexed first.
inputs first cause shape errors as the matrixs don't work properly
1d array = vector
2d array = matrix

output = weight * input + bias
similar to math y = mx + b
'''
