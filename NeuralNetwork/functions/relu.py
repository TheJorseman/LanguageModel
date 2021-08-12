import numpy as np

from NeuralNetwork.functions.activationfunctions import ActivationF

class ReLU(ActivationF):
  """
  Se define la función ReLU y su derivada.
  """
  def forward(self, x):
    return x * (x > 0)

  def derivate(self, x):
    return np.heaviside(x,1)
