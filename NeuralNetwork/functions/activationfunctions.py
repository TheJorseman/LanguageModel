from NeuralNetwork.module import Module

class ActivationF(Module):
  """
  Se define la clase base para las funciones de activación.
  """
  def __init__(self):
    super(ActivationF,self).__init__()
    self.x = None

  # Si se llama a backward implicitamente se llama a su derivada con los parametros con los que fue llamado.
  def backward(self):
    return self.derivate(*self.args)

  def derivate(self, x):
    raise NotImplementedError("Not Derivate in function {}".format(self.__class__.__name__))

