from gensim.models import Word2Vec
import pickle
from pprint import pprint
import numpy as np

with open('label','rb') as mysavedata:
    rate= pickle.load(mysavedata)
with open('tokens','rb') as mysavedata:
    tokens = pickle.load(mysavedata)
model = Word2Vec.load('256gpust1s')

pprint(model.most_similar('worst'))
