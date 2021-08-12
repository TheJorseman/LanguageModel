import numpy as np

from NeuralNetwork.module import Module

class ECMLoss(Module):
  """
  Se define la función forward para el error cuadratico medio con dos clases. (No utilizado)
  """
  def forward(self, input, target):
    self.input = input
    self.target = target
    return np.power(input-target, 2) / 2
  
  def backward(self):
    return self.derivate(self.input, self.target)

  def derivate(self, input, target):
    return (input - target)