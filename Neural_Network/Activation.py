from Activation_layer import Activation
import numpy as np

class TanH(Activation):
    def __init__(self, activation, activation_prime):
        tanh = lambda x : np.tanh(x)
        tanh_prime = lambda x : 1 - ((np.tanh(x)**2))
        super().__init__(tanh, tanh_prime)