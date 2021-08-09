import numpy as np

from NeuralNetwork.functions.activationfunctions import ActivationF

class ReLU(ActivationF):
  def forward(self, x):
    return x * (x > 0)

  def derivate(self, x):
    return np.heaviside(x,1)

