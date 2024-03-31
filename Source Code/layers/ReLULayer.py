import numpy as np

class ReLU():
    
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, gradout):
        new_grad = gradout.copy()
        new_grad[self.x < 0] = 0.
        return new_grad