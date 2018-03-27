import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import nltk
import re
from pprint import pprint

data = pd.read_excel('Womens Clothing E-Commerce Reviews.xlsx')




data = np.array(data)

text = 'ST '+data[:,3]
rate = data[:,4]
pprint(text[1])






tokens=[]


for i in range(len(text)):
    token = nltk.word_tokenize(text[i])
    tokens.append(token)

with open('tokens','wb') as mysavedata:
    pickle.dump(tokens, mysavedata)
with open('label','wb') as mysavedata:
    pickle.dump(rate, mysavedata)
from pprint import pprint
