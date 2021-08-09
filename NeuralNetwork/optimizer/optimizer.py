import numpy as np

class Optimizer(object):
  def __init__(self, model, loss, lr=0.005):
    self.model = model
    self.parameters = model.parameters()
    self.loss = loss
    self.lr = lr
    
  def zero_grad(self):
    return

  def step(self):
    return
