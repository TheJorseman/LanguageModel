import numpy as np

from NeuralNetwork.functions.activationfunctions import ActivationF

class Sigmoid(ActivationF):
  def forward(self, x):
    return 1/(1 + np.exp(-x))
  
  def derivate(self, x):
    sigma = self.forward(x)
    return sigma * (1 - sigma)

