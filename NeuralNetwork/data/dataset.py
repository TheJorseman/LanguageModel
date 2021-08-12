class Dataset():
  # Crea la estructura para el dataset.
  def __init__(self, *args):
    self.args = args
  # Método para obtener un item (Dataset[0] por ejemplo)
  def __getitem__(self, i):
    return []
  # Metodo para obtener la longitud
  def __len__(self):
    return 0
    