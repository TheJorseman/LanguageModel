import numpy as np

from NeuralNetwork.module import Module

class Linear(Module):
  """
  Se define la capa lineal o fully conected.
  """
  def __init__(self, input_size, output_size, norm=1, bias=True):
    """
    Constructor.
    Args:
      input_size    (int) : Dimension de entrada de la capa.
      output_size   (int) : Dimension de salida de la capa.
      norm          (int) : Factor de normalización para que no haya desvanecimiento o explocion de los pesos al realizar GD.
      bias          (bool): Si la capa requiere de bias o no.
    """
    super(Linear,self).__init__()
    self.is_bias = bias
    self.w = np.random.rand(input_size, output_size)*1*(1/np.sqrt(input_size))
    bias_function = np.ones if bias else np.zeros
    self.b = bias_function(output_size)

  def forward(self, x):
    # Se aplica el tensordot para manejar estructuras como (Batch Size, *, Vector)
    return np.tensordot(x,self.w, axes=(-1,0)) + self.b

  def backward(self):
    return self.args[0]

  def update_values(self, dw, db):
    # Se actualizan los valores segun las derivadas.
    self.w -= dw
    self.b -= db * int(self.is_bias) 