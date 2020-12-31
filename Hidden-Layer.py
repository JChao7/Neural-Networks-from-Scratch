#Hidden Layer Activation Functions
import numpy as np

spiral_data = np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, i))
    
'''
same thing as append(max)
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
'''

print(output)

#100 feature sets with 3 classes
X, y = spiral_data(100, 3)   

#first layer from input data
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #overflow protection
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablilities

dense1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(5,3)
activation2 = Activation_Softrmax()

dense1.forward(X) #pass data
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
#negative numbers turn into 0

print(activation2.output[:5]) #batch the first 5 outputs


'''
notes:
X - feature sets
y - targets / classifications

Step function - 0 or 1 answers, not detailed as we don't know how close we are to the numbers
Sigmoid function - more granular and is more preferable for later oppimizations and training the neural network
    vanishing gradiant problem -
Rectified Linear Activation function(ReLU) - granular, fast, output x if greater than n. Else output 0. Similar to step function + Sigmoid
    bias offsets the function horizontally (x-axis)
    2nd bias offsets the function vertically (y-axis)
    negative weight flip the function, and determins where the function deactivates

each neuron has a small role in the overall effect of the operation
area of effect comes into play when both connected neurons are activated
'''
