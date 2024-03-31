class CNN():
    
    def __init__(self, blocks: list):
        self.blocks = blocks
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block.forward(x)
  
        return x

    def backward(self, gradout):
        
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
            
        return gradout