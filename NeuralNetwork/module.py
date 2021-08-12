import numpy as np

np.random.seed(0)

class Module(object):
  """
  Clase base para crear los modulos de la red neuronal.
  Inspirado en la clase Module de Pytorch.
  """
  def __init__(self):
    # Elementos que contiene el modulo.
    self.module_list = []
    # Argumentos que se le pasan al forward
    self.args = None
    # Si el elemento requiere de backward.
    self.apply_backward = True

  def forward(self):
    return

  def backward(self):
    return

  def __call__(self, *args):
    """
    Esta funcion nos sirve para hacer un wrap cuando llaman al modulo. de esta manera resulta igual hacer
    module.forward() que module().
    """
    self.args = args
    self.output = self.forward(*args)
    return self.output

  def parameters(self):
    # Obtiene los parametros del modulo
    return list(filter(lambda m: m.apply_backward == True, self.module_list))