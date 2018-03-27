from gensim.models import Word2Vec
import pickle
from pprint import pprint
import numpy as np
import tensorflow as tf

with open('label','rb') as mysavedata:
    rate= pickle.load(mysavedata)
with open('tokens','rb') as mysavedata:
    tokens = pickle.load(mysavedata)
model = Word2Vec.load('128gpu')


tok=[]
for i in range(len(tokens)):
    c = []
    for s in tokens[i]:

        if s in model.wv.vocab:
            c.append(s)

    tok.append(c)

embeded=[]
for i in range(len(tok)):

    embed = model[tok[i]]
    embeded.append(embed)











embeded = embeded[:1000]



embeded = np.array(embeded)
pprint(embeded.reshape([-1]))

X= tf.placeholder(tf.float32, [None,None,128])

a = X

with tf.Session() as sess:
    s =sess.run(a,feed_dict={X:embeded})
    pprint(s)








