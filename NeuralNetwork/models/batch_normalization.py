import numpy as np

from NeuralNetwork.module import Module

class BatchNorm1d(Module):
  """
  Se define el BatchNormalization, esto se definió para utilizar la ReLU ya que habia una exploción del gradiente.
  """
  def __init__(self, eps=0.00001, gamma=1, beta=0):
    super(BatchNorm1d, self).__init__()
    self.apply_backward = False
    self.gamma = gamma
    self.beta = beta
    self.eps = eps

  def forward(self, x):
    mean = x.mean()
    var = x.var()
    output = (x - mean)/(np.sqrt(var + self.eps))
    return (output*self.gamma) + self.beta