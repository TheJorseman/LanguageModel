from tools.bos_eos import split_token, out_of_vocabulary_token

def get_index_vocab(vocabulary, token1, token2):
  default = vocabulary.get(out_of_vocabulary_token)
  return (vocabulary.get(token1, default), vocabulary.get(token2, default))

def get_bigrams(line, vocabulary):
  tokens = list(filter(lambda w: w != "", line.split(" ")))
  bigrams = []
  for i in range(len(tokens)-1):
    bigrams.append(get_index_vocab(vocabulary, tokens[i], tokens[i+1]))
  return bigrams

def create_bigrams(text, vocabulary):
  lines = text.split(split_token)
  sentences = []
  for line in lines:
    sentences += get_bigrams(line, vocabulary)
  return sentences