from collections import Counter

from tools.bos_eos import split_token
from tools.bos_eos import out_of_vocabulary_token

def get_types(text):
  tokens = text.split(" ")
  return dict(Counter(tokens))


def replace_freq_to_oov(vocabulary, freq, token_oov):
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
  vocabulary = get_types(text.replace(split_token, " "))
  vocabulary.pop('', None)
  return replace_freq_to_oov(vocabulary, freq, token_oov) 