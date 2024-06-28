from Layer import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, p:float) -> None:
        self.p = p

    def forward(self, input, training=True):
        if training:
            self.mask = np.random.binomial(1, self.p,size=input.shape)
            return (input * self.mask) / self.p
        else:
            return input
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask