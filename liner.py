import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import nltk
import re
from pprint import pprint
import tensorflow.contrib as tfc
import time
is_training = tf.placeholder(tf.bool)
step = 64
wdim = 256
dim = 256
Y = tf.placeholder(tf.float32,[None,5])
X = tf.placeholder(tf.float32,[None,64,wdim])
Z = tf.placeholder('float')

with open('embededst256','rb') as mysavedata:
    data11 = pickle.load(mysavedata)
with open('label','rb') as mysavedata:
    test11 = pickle.load(mysavedata)

max= 1000
line = 900


test11 =test11[:max]
data11 = data11[:max]
test11 = test11 -1
data = data11[:line ]
label = test11[:line ]
label=np.reshape(label,[-1,1])
datay = data11[line :]
labely = test11[line :]
labely=np.reshape(labely,[-1,1])

def pred(logic,label):
    prediction = tf.arg_max(logic, 1)
    lab = tf.arg_max(label, 1)
    is_correct = tf.reduce_mean(tf.cast(tf.equal(prediction, lab), dtype=tf.float32))
    return prediction,is_correct

def train(c,Y,z):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=c,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=z).minimize(cost)
    return cost,optimizer

def liner(lay2):
    lay2 = tf.reshape(lay2, [-1, step*wdim])
    W3 = tf.get_variable("W2", shape=[step*dim, 64], initializer=tfc.layers.xavier_initializer(step*dim,64))
    b1 = tf.get_variable('bw', shape=[64], dtype=tf.float32, initializer=tf.zeros_initializer(),
                        trainable=True)
    fw = tf.matmul(lay2, W3) + b1
    fii =tf.nn.leaky_relu(fw)

    W3 = tf.get_variable("Wq", shape=[64, 5], initializer=tfc.layers.xavier_initializer(64, 5))
    b = tf.get_variable('bq', shape=[5], dtype=tf.float32, initializer=tf.zeros_initializer(),
                        trainable=True)
    f = tf.matmul(fii, W3) + b
    fi = tf.nn.softmax(f)


    return f , fi


with tf.variable_scope("en")as scopes:

    ori, af=liner(X)
    cost =train(ori, Y, Z)
    pr,a= pred(af, Y)


label1 = tf.one_hot(label, depth=5)
labely1 = tf.one_hot(labely, depth=5)
label12 = tf.reshape(label1, [-1, 5])
labely12 = tf.reshape(labely1, [-1, 5])

v=0.0001
v1=0.00001

with tf.Session() as sess:
    save_path = "./save256111/model.ckpt"
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    label12 = sess.run(label12)
    labely12 = sess.run(labely12)

    start_time = time.time()
    for step in range(50001):
        if step <= 1000:
            p = sess.run(cost, feed_dict={X: data, Y: label12, is_training: True, Z: v})
            o, i = p
            print(step, o)
            if step % 50 == 0:
                testp = sess.run(pr, feed_dict={X: datay, Y: labely12, is_training: False})
                testY = np.array(labely)
                testa = sess.run(a, feed_dict={X: datay, Y: labely12, is_training: False})
                print(testa)

                import matplotlib.pyplot as plt

                plt.plot(testY + 1)
                plt.plot(testp + 1)
                plt.show()
                print("start_time", start_time)
                print("--- %s seconds ---" % (time.time() - start_time))
                saver.save(sess, save_path)
        else:
            p = sess.run(cost, feed_dict={X: data, Y: label12, is_training: True, Z: v1})
            o, i = p
            print(step, o)
            if step % 50 == 0:
                testp = sess.run(pr, feed_dict={X: datay, Y: labely12, is_training: False})
                testY = np.array(labely)
                testa = sess.run(a, feed_dict={X: datay, Y: labely12, is_training: False})
                print(testa)

                import matplotlib.pyplot as plt

                plt.plot(testY)
                plt.plot(testp)
                plt.show()
                print("start_time", start_time)
                print("--- %s seconds ---" % (time.time() - start_time))
                saver.save(sess, save_path)