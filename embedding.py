from gensim.models import Word2Vec
import pickle


with open('label','rb') as mysavedata:
    rate= pickle.load(mysavedata)
with open('tokens','rb') as mysavedata:
    tokens = pickle.load(mysavedata)


embedding_model = Word2Vec(tokens, size=100, window = 8, min_count=5, workers=4, iter=5, sg=1,alpha=0.05,sample=1e-4,negative=10)
embedding_model.save('100gpust1s')


