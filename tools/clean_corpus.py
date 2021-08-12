# Biblioteca para obtener los signos de puntuacion
import string
# Se importa SnowballStemmer para realizar el stemming
from nltk.stem import SnowballStemmer
# Se establece el stemmer y se configura para el español.
stemmer = SnowballStemmer('spanish')
#EOS_punct_add = {'.',';',':','?', '!', '...'}
EOS_punct = {'.'}
BOS_punct = set()
#TEMP_BOS = '-'
#BOS_punct = {'¿', '¡',TEMP_BOS}
ADDITIONAL = {';',':','?', '!', '...','¿', '¡','-'}
# Se construye el conjunto de signos de puntuación adicionalmente se añaden estos caracteres ya que en español si se usan.
exclude = set(string.punctuation + "<>" + string.digits ).union(ADDITIONAL) - EOS_punct - BOS_punct

def clean_punctuation_line(line):
  """
  Limpia de signos de puntuación de un string.
  Args:
    line (str): Texto que se quiere limpiar.
  Returns:
    str: Texto limpio.
  """   
  return ''.join(ch for ch in line if ch not in exclude)

def clean_line(line):
  """
  Limpia de signos de puntuación de un string.
  Args:
    line (str): Texto que se quiere limpiar.
  Returns:
    str: Texto limpio.
  """
  sentence = ''.join(ch for ch in line if ch not in exclude)
  sentence = sentence.replace("\xa0","")
  sentence = sentence.replace("\t"," ")
  sentence = sentence.replace("—fol. v→","")
  sentence = sentence.strip().lower()
  if len(sentence) < 2  or sentence == "":
    return ""
  return sentence + " "

def stemming_words(line):
  """
  Realiza el stemming de las palabras de un texto.
  Args:
      line (str): Texto que se quiere stemmizar.
  Returns:
      str: Texto limpio.
  """   
  words = line.split()
  return " ".join([stemmer.stem(word) for word in words]) + " "

def remove_consecutive_eos_punct(text):
  punct = zip(EOS_punct,EOS_punct)
  for p in punct:
    text = text.replace("".join(p), ".")
  return text

def clean_raw_corpus(raw_corpus_list, stem_words=True):
  """
  Limpia el corpus de lineas vacias, signos de puntuación y hace el stemming de cada palabra.
  Args:
      raw_corpus_list (list): Lista de strings que se quieren limpiar.
  Returns:
      str: Texto limpio.
  """   
  clean_str = ""
  for line in raw_corpus_list:
    new_line = line.strip()
    new_line = clean_line(new_line)
    if stem_words:
      new_line = stemming_words(new_line)
    clean_str += new_line
  #clean_str = remove_consecutive_eos_punct(clean_str)
  return clean_str.strip()