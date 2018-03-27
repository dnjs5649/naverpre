import pickle
from pprint import pprint
import numpy as np
import tensorflow as tf
import time



def xavier_initializer(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

with open('embeded1000','rb') as mysavedata:
    data11 = pickle.load(mysavedata)
with open('label','rb') as mysavedata:
    test11 = pickle.load(mysavedata)
test11 =test11[:1000]
test11 = test11 -1
data = data11[:900]
label = test11[:900]
label=np.reshape(label,[-1,1])
datay = data11[900:]
labely = test11[900:]
labely=np.reshape(labely,[-1,1])

from pprint import pprint



step =32
dim=128

Y = tf.placeholder(tf.float32,[None,5])
X = tf.placeholder(tf.float32,[None,32,128])
Z = tf.placeholder('float')

import tensorflow.contrib as tfc

is_training = tf.placeholder(tf.bool)

def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=0.7,
                      var_scope_name="conv_layer"):

    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * dropout / (kernel_size * in_dim))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)


        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs

def linear_mapping_weightnorm(inputs, out_dim, in_dim=None, dropout=0.7, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()
        input_shape_tensor = tf.shape(inputs)

        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                dropout * 1.0 / int(input_shape[-1]))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])


        scaler = tf.div(g, tf.norm(V, axis=0))
        inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])
        return inputs

def gated_linear_units(inputs):
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 3
  input_pass = inputs[:,:,0:int(input_shape[2]/2)]
  input_gate = inputs[:,:,int(input_shape[2]/2):]
  input_gate = tf.sigmoid(input_gate)
  return tf.multiply(input_pass, input_gate)

def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict,is_training):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        res_inputs = next_layer

        if layer_idx == 1:
            next_layer = tf.contrib.layers.dropout(
                inputs=next_layer,
                keep_prob=dropout_dict['hid'], is_training=is_training)

            next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="SAME", dropout=dropout_dict['hid'],
                                       var_scope_name="conv_layer_" + str(layer_idx))


            next_layer = gated_linear_units(next_layer)
            next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)


        elif layer_idx == 2:
            next_layer = tf.contrib.layers.dropout(
                inputs=next_layer,
                keep_prob=dropout_dict['hid'], is_training=is_training)

            next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                           kernel_size=kwidths_list[layer_idx], padding="SAME",
                                           dropout=dropout_dict['hid'],
                                           var_scope_name="conv_layer_" + str(layer_idx))


            next_layer = gated_linear_units(next_layer)
            next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)


        elif layer_idx == 3:
            next_layer = tf.contrib.layers.dropout(
                inputs=next_layer,
                keep_prob=dropout_dict['hid'], is_training=is_training)

            next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                           kernel_size=kwidths_list[layer_idx], padding="SAME",
                                           dropout=dropout_dict['hid'],
                                           var_scope_name="conv_layer_" + str(layer_idx))


            next_layer = gated_linear_units(next_layer)
            next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)


    return next_layer

def encode( inputs):
    embed_size = inputs.get_shape().as_list()[-1]

    with tf.variable_scope("encoder_cnn"):
        next_layer = inputs
        if 4 > 0:
            nhids_list = [512,512,512,512]
            kwidths_list = [3,3,3,3]
            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0],
                                                   dropout=0.7,
                                                   var_scope_name="linear_mapping_before_cnn")
            next_layer = conv_encoder_stack(next_layer, nhids_list, kwidths_list,
                                            {'src': 0.7,
                                             'hid': 0.7}, is_training)

            next_layer = linear_mapping_weightnorm(next_layer, embed_size, var_scope_name="linear_mapping_after_cnn")

        cnn_c_output = (next_layer + inputs) * tf.sqrt(0.5)

    final_state = tf.reduce_mean(cnn_c_output, 1)

    return  next_layer,final_state,cnn_c_output

def liner(lay2):
    lay2 = tf.reshape(lay2, [-1, step*dim])
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

def train(c,Y,z):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=c,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=z).minimize(cost)
    return cost,optimizer

def final(X,Y,z):

    li2=encode(X)
    l,x,c =li2
    f, fi=liner(l)
    cost=train(f,Y,z)
    return cost ,fi


def pred(logic,label):
    prediction = tf.arg_max(logic, 1)
    lab = tf.arg_max(label, 1)
    is_correct = tf.reduce_mean(tf.cast(tf.equal(prediction, lab), dtype=tf.float32))
    return prediction,is_correct
v=0.0001
v1=0.00005

with tf.variable_scope("en")as scopes:
    cost,h=final(X,Y,Z)
    tf.get_variable_scope().reuse_variables()
    li=encode(X)
    cv,x,c=li
    q,w =liner(cv)
    out = w
    pr,a=pred(out,Y)
label1 = tf.one_hot(label, depth=5)
labely1 = tf.one_hot(labely, depth=5)
label12 = tf.reshape(label1, [-1, 5])
labely12 = tf.reshape(labely1, [-1, 5])


with tf.Session() as sess:



    save_path = "./save1000/model.ckpt"
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    label12=sess.run(label12)
    labely12=sess.run(labely12)

    start_time = time.time()
    for step in range(50001):
        if step <= 1000:
            p=sess.run(cost,feed_dict={X:data,Y:label12,is_training:True,Z:v})
            o, i = p
            print(step,o)
            if step % 50 == 0:
                testp = sess.run(pr, feed_dict={X: datay, Y:labely12, is_training: False})
                testY = np.array(labely)
                testa = sess.run(a, feed_dict={X: datay, Y: labely12, is_training: False})
                print(testa)

                import matplotlib.pyplot as plt

                plt.plot(testY+1)
                plt.plot(testp+1)
                plt.show()
                print("start_time", start_time)
                print("--- %s seconds ---" % (time.time() - start_time))
                saver.save(sess, save_path)
        else:
            p=sess.run(cost,feed_dict={X:data,Y:label12,is_training:True,Z:v1})
            o, i = p
            print(step,o)
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