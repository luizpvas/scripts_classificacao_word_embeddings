#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import gensim
from gensim.models.wrappers import FastText
import numpy as np

# Dimensão que os vetores foram treinados
dim_vec = 300

# Tópicos escolhidos: Medicina, Engine gráfica, Baseball
topics = ['sci.med', 'comp.graphics', 'rec.sport.baseball', 'rec.motorcycles', 'talk.religion.misc']
# topics = ['sci.med']

docs = {}

# Carregar o modelo pré-treinado
# model = FastText.load_fasttext_format('wiki.simple/wiki.simple')

total_docs = 0
total_words = 0

with open(sys.argv[1]) as file:
  for line in file:
    parts = line.split('\t')
    topic = parts[0]
    doc = parts[1]

    total_docs += 1
    total_words += len(doc.split(' '))

    if topic in docs:
      docs[topic].append(doc)
    else:
      docs[topic] = [doc]

print("Media: {}".format(total_words / total_docs))
sys.exit(1)

print('@RELATION word2vec')

for i in range(300):
  print('@ATTRIBUTE dim{} NUMERIC'.format(i))

topics_str = '{' + ','.join(topics) + '}'
print('@ATTRIBUTE class {}'.format(topics_str))

print('@DATA')

for topic in topics:
  for doc in docs[topic]:
    total_words = 0

    doc_vec = np.zeros(dim_vec)

    # Para cada palavra no documento, nós vamos somar em um único vetor.
    for word in doc.split(' '):
      # As palavras que não existirem no modelo devem ser ignoradas.
      try:
        doc_vec += np.array(model.wv[word])
        total_words += 1
      except:
        pass

    # Agora é necessário dividir o vetor pela quantidade de palavras para
    # normalizar os valores
    doc_vec = doc_vec / total_words

    # Nesse ponto nós temos o vetor do documento
    attributes = doc_vec.tolist()
    attributes.append(topic)
    print(','.join(str(x) for x in attributes))

