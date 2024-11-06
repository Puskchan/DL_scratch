from Activation_layer import Activation
import numpy as np

class TanH(Activation):
    def __init__(self):
        tanh = lambda x : np.tanh(x)
        tanh_prime = lambda x : 1 - ((np.tanh(x)**2))
        super().__init__(tanh, tanh_prime)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x : np.maximum(0,x)
        relu_prime = lambda x : np.where(x > 0, 1, 0)
        super().__init__(relu, relu_prime)


class Leaky_ReLU(Activation):
    def __init__(self, a=0.01):
        leaky = lambda x : np.where(x>0, x, a*x)
        leaky_prime = lambda x : np.where(x>0,1,a) 
        super().__init__(leaky, leaky_prime)