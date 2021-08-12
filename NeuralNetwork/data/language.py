import numpy as np

from NeuralNetwork.data.dataset import Dataset

class LanguageDataset(Dataset):
  """
  Se crea el dataset de lenguaje, se le pasan los bigramas de train y test ademas del vocabulario.
  """
  def __init__(self, vocabulary, bigrams_train, bigrams_test):
    """
    Constructor:
    Args:
      vocabulary    (dict): Vocabulario con la estructura {token: índice}
      bigrams_train (list): Lista de tuplas que contienen los bigramas de entrenamiento.
      bigrams_train (list): Lista de tuplas que contienen los bigramas de prueba.
    """
    self.vocabulary = vocabulary
    # Se crea aqui la matriz one hot para ahorrar memoria y solo utilizar una matriz.
    self.one_hot_matrix = np.identity(len(vocabulary))
    self.bigrams_train = BigramDataset(bigrams_train, self.one_hot_matrix)
    self.bigrams_test = BigramDataset(bigrams_test, self.one_hot_matrix)
  
  def get_train_dataset(self):
    return self.bigrams_train
  
  def get_test_dataset(self):
    return self.bigrams_test


class BigramDataset(LanguageDataset):
  """
  Clase que almacena todos los bigramas y los procesa para su iteración
  """
  def __init__(self, bigrams, one_hot_matrix):
    self.bigrams = bigrams
    self.one_hot_matrix = one_hot_matrix

  def __getitem__(self, i):
    bi = self.bigrams[i]
    mat = self.one_hot_matrix
    return {'input': np.array([ mat[bi[0]] ]), 'output': np.array([ mat[bi[1]] ])}

  def __iter__(self):
    mat = self.one_hot_matrix
    return iter([{'input': np.array([ mat[bi[0]] ]), 'output': np.array([ mat[bi[1]] ])} for bi in self.bigrams])

  def __len__(self):
    return len(self.bigrams)