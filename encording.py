from gensim.models import Word2Vec
import pickle
from pprint import pprint
import numpy as np

with open('label','rb') as mysavedata:
    rate= pickle.load(mysavedata)
with open('tokens','rb') as mysavedata:
    tokens = pickle.load(mysavedata)
model = Word2Vec.load('100gpust1s')

dim = 100
wdim = 100
tok=[]
for i in range(len(tokens)):
    c = []
    for s in tokens[i]:

        if s in model.wv.vocab:
            c.append(s)

    tok.append(c)

for i in range(len(rate)):
    if len(tok[i]) > dim:
        tok[i] = tok[i][0:dim]

embeded=[]
for i in range(len(tok)):
    if len(tok[i])==dim:
        embed = model[tok[i]]
        embeded.append(embed)
    else:
        embed = list(model[tok[i]])
        for a in range(dim-len(tok[i])):
            embed.append(np.zeros(wdim))
        embeded.append(embed)

pprint(len(embeded))
embeded = np.array(embeded[:20000])

with open('embededst100','wb') as mysavedata:
    pickle.dump(embeded, mysavedata)









