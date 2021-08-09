import numpy as np

from NeuralNetwork.functions.activationfunctions import ActivationF

class Softmax(ActivationF):
  def forward(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(-1, keepdims=True)
  
  def derivate(self, x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

