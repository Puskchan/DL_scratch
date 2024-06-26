from Dense_layer import Dense
from Activation import TanH
from Loss import mse, mse_prime
import numpy as np

X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
Y = np.reshape([[0],[1],[1],[0]], (4,1,1))

network = [
    Dense(2,3),
    TanH(),
    Dense(3,1),
    TanH()
]

epoches = 10000
learning_rate = 0.1

for e in range(epoches):
    error = 0
    for x,y in zip(X,Y):
        output = x
        for layer in network:
            output = layer.forward(output)

        error += mse(y,output)

        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    if (e+1) % 1000 == 0:
        print(f'{e+1} {epoches}, error: {error}')

output = [[1],[0]]
for layer in network:
    output = layer.forward(output)
if output > 0.5: print(1) 
else: print(0)