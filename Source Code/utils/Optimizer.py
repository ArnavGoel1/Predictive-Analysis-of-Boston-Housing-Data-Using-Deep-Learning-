from layers.FullyConnectedLayer import FullyConnectedLayer
from layers.CNN import CNN

class Optimizer():
    
    def __init__(self, lr, nn: CNN):
        self.lr = lr
        self.nn = nn
        
    def step(self):
        
        for block in self.nn.blocks:
            if block.__class__ == FullyConnectedLayer:
                block.W = block.W - self.lr * block.dW
                block.b = block.b - self.lr * block.db