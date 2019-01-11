import gensim

docs = {}

with open('20ng-train-all-terms.txt') as file:
  for line in file:
    parts = line.split('\t')
    topic = parts[0]
    doc = parts[1]

    if topic in docs:
      docs[topic].append(doc)
    else:
      docs[topic] = [doc]

for topic, docs in docs.items():
  print('%s - %s' % (topic, len(docs)))
