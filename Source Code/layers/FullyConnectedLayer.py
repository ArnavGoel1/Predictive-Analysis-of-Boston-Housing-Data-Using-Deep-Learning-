import numpy as np
class FullyConnectedLayer():
    
    def __init__(self, input, output):
        self.W = (2 * np.random.rand(output, input) - 1) * (np.sqrt(6) / np.sqrt(input + output))
        self.b = (2 * np.random.rand(output) - 1) * (np.sqrt(6) / np.sqrt(input + output))
        
    def forward(self, x): 
        self.x = x 
        return x @ self.W.T + self.b

    def backward(self, gradout):
        self.dW = gradout.T @ self.x
        self.db = gradout.sum(0)
        return gradout @ self.W
    