import math

softmax_output = [0.7, 0.1, 0.2] #example output

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]*target_output[0]) +
         math.log(softmax_output[1]*target_output[1]) +
         math.log(softmax_output[2]*target_output[2]))

print(loss)

'''
Calculating Loss with Categorical Cross-Entropy
---

Outputs calculate a probability distribution. (Confidence Score)
Loss fuction - means abosulte error + regression - avg of distances of perdiction vs perdiction value. Lower score is more accurate.

Loss and Confidence scores are opposite relations, higher confidence, lower loss score

Categorical Cross-Entropy - 
One hot coding - vector that is n classes long and filled with 0 except for last value is 1. The 1 is the one hot vector.
    Label 0 puts the vector 1 at the beginning. Label 1 puts the hot vector in the 2nd spot, etc.
Logogeritum base e(Euler's number, 2.718...) is also natural log
Log is solving for x in the equation e ** x = b
'''
