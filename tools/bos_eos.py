bos_token = "<BOS>"
eos_token = "<EOS>"
split_token = "<SPLIT>"
out_of_vocabulary_token = '<OOV>'

EOS_punct = {'.'}
BOS_punct = set()

def insert_bos_eos(corpus):
  """
  Función para insertar tokens de inicio y de fin de cadena. La separación se hace con respecto al punto . y se añade 
  un token para hacer el split entre las oraciones.
  Args:
      corpus (str): Corpus a utilizar.
  Returns:
      str: Texto con EOS, BOS y OOV
  """
  new_corpus = corpus
  replace_sentence = " " + eos_token + split_token + bos_token + " "
  for eos in EOS_punct:
    new_corpus = new_corpus.replace(eos, replace_sentence)
  if new_corpus.endswith(eos_token + " "):
    new_corpus = new_corpus[:-len(replace_sentence)]
  return bos_token + " "+ new_corpus + " " + eos_token + " " + out_of_vocabulary_token

def get_reduced_corpus(corpus, length):
  """
  Función para reducir el tamaño del corpus.
  Args:
      corpus (str): Corpus a reducir.
      length (int): Cuantas oraciones se quiere.
  Returns:
      str: Texto reducido.
  """  
  reduced = corpus.split(split_token)[:length]
  return split_token.join(reduced)