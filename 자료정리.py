import pickle
from pprint import pprint
import numpy as np
import tensorflow as tf
import time
from gensim.models import Word2Vec
model = Word2Vec.load('256gpust')


def xavier_initializer(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)
Y = tf.placeholder(tf.float32,[None,5])
pprint([256]*4)
X = tf.placeholder(tf.float32,[None,1,128])
Z = tf.placeholder('float')

start1 = model['ST']
pprint(start1)
start1 = np.tile(start1, 900)

start1 = np.reshape(start1,[900,1,256])
pprint(start1.shape)