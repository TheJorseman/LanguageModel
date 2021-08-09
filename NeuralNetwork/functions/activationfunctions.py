from NeuralNetwork.module import Module

class ActivationF(Module):
  def __init__(self):
    super(ActivationF,self).__init__()
    self.x = None

  def backward(self):
    return self.derivate(*self.args)

  def derivate(self, x):
    raise NotImplementedError("Not Derivate in function {}".format(self.__class__.__name__))

