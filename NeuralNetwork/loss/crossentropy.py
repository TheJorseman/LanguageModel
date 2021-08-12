import numpy as np

from NeuralNetwork.module import Module

class CrossEntropyLoss(Module):
  """
  Se define el Cross Entropy Loss
  """
  def forward(self, input, target):
    x = -np.log(self.entropy(input, target))
    return np.array([x])
  # Función para calcular la derivada. Si tiene softmax entonces se realiza la derivada simple.
  def backward(self, softmax=False):
    return self.derivate(*self.args, softmax=softmax)

  def derivate(self, input, target, softmax=False):
    if softmax:
      out = self.derivate_softmax(input, target)
      return out
    # La derivada cuando la salida de la red NO es una Softmax No esta definida.
    return input - target
  # Se calcula la entropia
  def derivate_softmax(self, input, target):
    return input - target
  # Se calcula la entropia
  def entropy(self, input, target):
    x = np.tensordot(input, target, axes=(-1,-1))
    return np.diag(x)