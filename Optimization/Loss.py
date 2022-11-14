import numpy as np

class CrossEntropyLoss(object):
    
    def __init__(self):
        pass
        
    def forward(self, prediction_tensor, label_tensor):
        self.lastIn = prediction_tensor
        y_hat = prediction_tensor
        y = label_tensor
        loss = -np.sum(y * np.log(y_hat + np.finfo(float).eps))
        return loss
    
    def backward(self, label_tensor):
        return -(label_tensor / (self.lastIn + np.finfo(float).eps))

class L2Loss:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)