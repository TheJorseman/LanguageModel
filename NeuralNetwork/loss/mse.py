import numpy as np

from NeuralNetwork.module import Module

class MSELoss(Module):
  """
  Se define la función forward para el error medio cuadratico. (No utilizado)
  """
  def forward(self, input, target):
    return np.power(input-target, 2).mean(axis=1)
