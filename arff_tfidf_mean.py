#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import gensim
from gensim.models.wrappers import FastText
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np

# Dimensão que os vetores foram treinados
dim_vec = 300

# Tópicos escolhidos: Medicina, Engine gráfica, Baseball
topics = [
  'comp.graphics',
  'comp.os.ms-windows.misc',
  'comp.sys.ibm.pc.hardware',
  'comp.sys.mac.hardware',
  'comp.windows.x',
  'misc.forsale',
  'rec.autos',
  'rec.motorcycles',
  'rec.sport.baseball',
  'rec.sport.hockey',
  'talk.politics.misc',
  'talk.politics.guns',
  'talk.politics.mideast',
  'sci.crypt',
  'sci.electronics',
  'sci.med',
  'sci.space',
  'talk.religion.misc',
  'alt.atheism',
  'soc.religion.christian'
]

docs = {}

all_docs = []

with open(sys.argv[1]) as file:
  for line in file:
    parts = line.split('\t')
    topic = parts[0]
    doc = parts[1]

    if topic in docs:
      docs[topic].append(doc)
      all_docs.append(doc.split(' '))
    else:
      docs[topic] = [doc]

dct = Dictionary(all_docs)
corpus = [dct.doc2bow(line) for line in all_docs]
model = TfidfModel(corpus)
# vector = model[corpus[0]]

# Carregar o modelo pré-treinado
word2vec = FastText.load_fasttext_format('wiki.simple/wiki.simple')

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

    # Calculate the weights
    tfidf_vector = model[dct.doc2bow(doc.split(' '))]
    for wid, weight in tfidf_vector:
      try:
        word = dct.get(wid)
        doc_vec += np.array(word2vec.wv[word] * weight)
        total_words += weight
      except:
        pass

    # Agora é necessário dividir o vetor pela quantidade de palavras para
    # normalizar os valores
    doc_vec = doc_vec / total_words

    # Nesse ponto nós temos o vetor do documento
    attributes = doc_vec.tolist()
    attributes.append(topic)
    print(','.join(str(x) for x in attributes))

