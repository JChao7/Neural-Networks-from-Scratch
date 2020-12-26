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




# BATCH + LAYER
#batches - parallel opperations we can run. Helps with generalizations
#inputs * weights

inputs = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1.0, 2.0, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2 #transpose

print(layer2_outputs)



#CONVERT LAYERS TO OBJECT

np.random.seed(0)

#input dataset
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

#creating hidden layer
class Layer_Dense:
    #initialize model
    #we can skip using transpose when we can flip the inputs and neurons during initiation 
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #using the shapes to create the weights
        self.biases = np.zeros((1, n_neurons))

    #class to go to the next layer    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5) #size of inputs (4) and size of neurons we want to have (anything you want)
layer2 = Layer_Dense(5,2) #only requirment is the output of layer1 is the input of layer2

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


'''
Notes:
weights go first because thats the first to be "indexed", we want weights to be indexed first.
inputs first cause shape errors as the matrixs don't work properly
1d array = vector
2d array = matrix

output = weight * input + bias
similar to math y = mx + b

batch size helps with generalization and the average line for the weights
transpose - swaps rows and columns to fix shape errors

when setting a weight, you want smaller numbers (eg. 0.1) as we want to keep the range between +1 and -1
if you don't keep the number withen that range, your computer will explode

bias tend to be set at 0 unless the weights are too low and your network is "dead." Then you should increase the bias
'''
