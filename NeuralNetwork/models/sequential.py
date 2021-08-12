from NeuralNetwork.module import Module

class Sequential(Module):
  """
  Se define la estructura de una red Secuencial. En donde se le pueden añadir capas y funciones de activación.
  """
  def __init__(self, *args):
    super(Sequential,self).__init__()
    self.module_list = list(args)

  def forward(self, x):
    # Itera la entrada sobre todos los módulos.
    output = x
    for module in self.module_list:
      output = module(output)
    return output