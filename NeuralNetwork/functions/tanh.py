
import numpy as np

from NeuralNetwork.functions.activationfunctions import ActivationF

class Tanh(ActivationF):
  """
  Se define la función Tanh y su derivada.
  """
  def forward(self, x):
    return np.tanh(x)

  def derivate(self, x):
    h = self.forward(x)
    return 1-h**2
