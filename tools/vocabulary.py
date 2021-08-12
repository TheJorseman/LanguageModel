from collections import Counter

from tools.bos_eos import split_token
from tools.bos_eos import out_of_vocabulary_token

def get_types(text):
  """
  Función para obtener los tipos del corpus.
  Args:
    text (str): Texto a obtener los tipos.
  Returns:
    Diccionario con los tipos y su frecuencia.
  """
  tokens = text.split(" ")
  return dict(Counter(tokens))


def replace_freq_to_oov(vocabulary, freq, token_oov):
  """
  Función para reemplazar los tipos que tienen frecuencia 'freq' a el token OOV
  Args:
    vocabulary (dic): Diccionario con los tipos y la frecuencia.
    freq       (int): Frecuencia de corte en donde se quiere reemplazar por OOV.
    token_oov  (str): Token por el que se quiere reemplazar
  Returns:
    Diccionario con los tipos
  """
  output_vocab = {}
  i = 0
  for k,v in vocabulary.items():
    if v <= freq or k==token_oov:
      if not output_vocab.get(token_oov,False):
        output_vocab[token_oov] = i
        i +=1
    else:
      output_vocab[k] = i
      i +=1
  return output_vocab

def create_vocabulary(text, token_oov=out_of_vocabulary_token, freq=1):
  """
  Función para crear el vocabulario. 
  Itera sobre todos los tipos y les asigna un indice.
  Args:
    text        (dic): Texto a procesar.
    token_oov   (int): Token por el que se quiere reemplazar.
    freq        (int): Frecuencia de corte en donde se quiere reemplazar por OOV.
  Returns:
    Diccionario con el vocabulario.
  """
  vocabulary = get_types(text.replace(split_token, " "))
  vocabulary.pop('', None)
  return replace_freq_to_oov(vocabulary, freq, token_oov) 