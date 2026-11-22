import numpy as np

inputs = np.array([1,0])
weights = np.array([5,-8])
bias = -2

score = np.dot(inputs,weights) + bias
print("score : ",score)

random_weights = np.random.randn(2) * 0.1
print("random weight: ",random_weights) 

days = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

scores = np.dot(days,weights) + bias

print("all scores : ",scores)
dec = scores > 0
print("Run : ",dec)

