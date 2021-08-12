from tools.bos_eos import split_token, out_of_vocabulary_token

def get_index_vocab(vocabulary, token1, token2):
  """
  Función para obtener el indice de una palabra en el vocabulario.
  Args:
    vocabulary    (dic): Vocabulario que contiene la estructura {token:indice}
    token1        (str): Token por el que se quiere reemplazar.
    token2        (str): Token por el que se quiere reemplazar.
  Returns:
    tuple:  tupla con los indices del vocabulario.
  """
  default = vocabulary.get(out_of_vocabulary_token)
  return (vocabulary.get(token1, default), vocabulary.get(token2, default))

def get_bigrams(line, vocabulary):
  """
  Obtiene los bigramas de una linea (str) u oracion.
  Args:
    line        (str): linea a la cual se le calculan los bigramas.
    vocabulary  (dic): Vocabulario que contiene la estructura {token:indice}
  Returns:
    list:  lista con tuplas que contienen los bigramas
  """
  tokens = list(filter(lambda w: w != "", line.split(" ")))
  bigrams = []
  for i in range(len(tokens)-1):
    bigrams.append(get_index_vocab(vocabulary, tokens[i], tokens[i+1]))
  return bigrams

def create_bigrams(text, vocabulary):
  """
  Obtiene los bigramas de un corpus.
  Args:
    text        (str): Texto al cual se le calculan los bigramas.
    vocabulary  (dic): Vocabulario que contiene la estructura {token:indice}
  Returns:
    list:  lista con tuplas que contienen los bigramas
  """
  lines = text.split(split_token)
  sentences = []
  for line in lines:
    sentences += get_bigrams(line, vocabulary)
  return sentences