import numpy as np
class MSELoss:
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return np.mean((pred - true) ** 2)

    def backward(self):
        return 2 * (self.pred - self.true) / self.true.size

    def __call__(self, pred, true):
        return self.forward(pred, true)