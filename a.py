import pickle
from pprint import pprint
import numpy as np
import tensorflow as tf
import time
from gensim.models import Word2Vec
model = Word2Vec.load('100gpust1s')
hid = 256
elayer = 2
wdim = 100
dlayer = 2
step =100
def xavier_initializer(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)
Y = tf.placeholder(tf.float32,[None,5])
X = tf.placeholder(tf.float32,[None,step,wdim])
Z = tf.placeholder('float')


with open('embededst100','rb') as mysavedata:
    data11 = pickle.load(mysavedata)
with open('label','rb') as mysavedata:
    test11 = pickle.load(mysavedata)

max= 20000
line = 19000


test11 =test11[:max]
data11 = data11[:max]
test11 = test11 -1
data = data11[:line ]
label = test11[:line ]
label=np.reshape(label,[-1,1])
datay = data11[line :]
labely = test11[line :]
labely=np.reshape(labely,[-1,1])

from pprint import pprint




dim=256



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
            nhids_list = [hid] * elayer
            kwidths_list = [3] * elayer
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
    lay2 = tf.reshape(lay2, [-1, wdim*5])
    W3 = tf.get_variable("W21", shape=[wdim*5, 256], initializer=tfc.layers.xavier_initializer(wdim*5,256))
    b1 = tf.get_variable('bw', shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer(),
                        trainable=True)
    fw = tf.matmul(lay2, W3) + b1
    fii =tf.nn.leaky_relu(fw)

    W3 = tf.get_variable("Wq", shape=[256, 5], initializer=tfc.layers.xavier_initializer(256, 5))
    b = tf.get_variable('bq', shape=[5], dtype=tf.float32, initializer=tf.zeros_initializer(),
                        trainable=True)
    f = tf.matmul(fii, W3) + b
    fi = tf.nn.softmax(f)


    return f , fi

def train(c,Y,z):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=c,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=z).minimize(cost)
    return cost,optimizer

def final(X,Y,z,A):
    li = encode(X)
    cv, x, c = li
    xxx = infer_block(cv, c, A)
    f, fi = liner(xxx)
    cost=train(f, Y, z)
    return cost ,fi


def pred(logic,label):
    prediction = tf.arg_max(logic, 1)
    lab = tf.arg_max(label, 1)
    is_correct = tf.reduce_mean(tf.cast(tf.equal(prediction, lab), dtype=tf.float32))
    return prediction,is_correct
v=0.00003
v1=0.00001


##################################################










def conv_decoder_stack(target_embed, nl,cc, inputs, nhids_list, kwidths_list, dropout_dict, mode):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout_dict['hid'],
                                                   var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=dropout_dict['hid'],
            is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)
        # special process here, first padd then conv, because tf does not suport padding other than SAME and VALID
        next_layer = tf.pad(next_layer, [[0, 0], [kwidths_list[layer_idx] - 1, kwidths_list[layer_idx] - 1], [0, 0]],
                            "CONSTANT")

        next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="VALID",
                                       dropout=dropout_dict['hid'], var_scope_name="conv_layer_" + str(layer_idx))
        '''
        next_layer = tf.contrib.layers.conv2d(
            inputs=next_layer,
            num_outputs=nout*2,
            kernel_size=kwidths_list[layer_idx],
            padding="VALID",   #should take attention, not SAME but VALID
            weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4 * dropout_dict['hid'] / (kwidths_list[layer_idx] * next_layer.get_shape().as_list()[-1]))),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=None,
            scope="conv_layer_"+str(layer_idx))
        '''
        layer_shape = next_layer.get_shape().as_list()kt
        assert len(layer_shape) == 3
        # to avoid using future information
        next_layer = next_layer[:, 0:-kwidths_list[layer_idx] + 1, :]

        next_layer = gated_linear_units(next_layer)

        # add attention
        # decoder output -->linear mapping to embed, + target embed,  query decoder output a, softmax --> scores, scores*encoder_output_c-->output,  output--> linear mapping to nhid+  decoder_output -->
        att_out = make_attention(target_embed,nl,cc, next_layer, layer_idx)
        next_layer = (next_layer + att_out) * tf.sqrt(0.5)

        # add res connections
        next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
    return next_layer



def make_attention(target_embed, nl,cc, decoder_hidden, layer_idx):
    with tf.variable_scope("attention_layer_" + str(layer_idx)):
        embed_size = target_embed.get_shape().as_list()[-1]  # k
        dec_hidden_proj = linear_mapping_weightnorm(decoder_hidden, embed_size,
                                                    var_scope_name="linear_mapping_att_query")  # M*N1*k1 --> M*N1*k
        dec_rep = (dec_hidden_proj + target_embed) * tf.sqrt(0.5)

        encoder_output_a = nl
        encoder_output_c = cc  # M*N2*K

        att_score = tf.matmul(dec_rep, encoder_output_a, transpose_b=True)  # M*N1*K  ** M*N2*K  --> M*N1*N2
        att_score = tf.nn.softmax(att_score)

        length = tf.cast(tf.shape(encoder_output_c), tf.float32)
        att_out = tf.matmul(att_score, encoder_output_c) * length[1] * tf.sqrt(
            1.0 / length[1])  # M*N1*N2  ** M*N2*K   --> M*N1*k

        att_out = linear_mapping_weightnorm(att_out, decoder_hidden.get_shape().as_list()[-1],
                                            var_scope_name="linear_mapping_att_out")
    return att_out


def conv_block(nl,cc, input_embed, is_train=True):
    with tf.variable_scope("decoder_cnn"):
        next_layer = input_embed
        if 4 > 0:
            nhids_list = [hid] * dlayer
            kwidths_list = [3] * dlayer

            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0],
                                                   dropout=0.9,
                                                   var_scope_name="linear_mapping_before_cnnde")

            next_layer = conv_decoder_stack(input_embed,nl,cc, next_layer, nhids_list, kwidths_list,
                                            {'src': 0.9,
                                             'hid': 0.9},is_training)

            next_layer = linear_mapping_weightnorm(next_layer[:, -1:, :], wdim,
                                                   var_scope_name="linear_mapping_after_cnnde")


    return next_layer

def conv_block1(nl,cc, input_embed, is_train=True):
    with tf.variable_scope("decoder_cnn",reuse=True):
        next_layer = input_embed
        if 4 > 0:
            nhids_list = [hid] * dlayer
            kwidths_list = [3] * dlayer

            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0],
                                                   dropout=0.9,
                                                   var_scope_name="linear_mapping_before_cnnde")

            next_layer = conv_decoder_stack(input_embed,nl,cc, next_layer, nhids_list, kwidths_list,
                                            {'src': 0.9,
                                             'hid': 0.9},is_training)

            next_layer = linear_mapping_weightnorm(next_layer[:, -1:, :],wdim,
                                                   var_scope_name="linear_mapping_after_cnnde")


    return next_layer




P = tf.placeholder(tf.float32,[None,None,wdim])





def infer_block(v,b,n):
    a=n
    cvs=conv_block(v,b,n)
    co1 = tf.concat([a,cvs],1)
    coq = conv_block1(v,b,co1)
    co2 = tf.concat([co1,coq],1)
    coq1 = conv_block1(v, b, co2)
    co3 = tf.concat([co2,coq1],1)
    coq2 = conv_block1(v, b, co3)
    co4 = tf.concat([co3,coq2],1)


    return co4






with tf.variable_scope("en")as scopes:

    cost,h=final(X,Y,Z,P)
    tf.get_variable_scope().reuse_variables()
    li=encode(X)
    cv,x,c=li
    afa=infer_block(cv,c,P)
    fer, af=liner(afa)
    pr, a = pred(af, Y)



label1 = tf.one_hot(label, depth=5)
labely1 = tf.one_hot(labely, depth=5)
label12 = tf.reshape(label1, [-1, 5])
labely12 = tf.reshape(labely1, [-1, 5])




start11 = model['ST']

start11 = np.tile(start11, max)
start11 = np.reshape(start11,[max,1,wdim])

start1 = start11[line:]
start = start11[:line]
with tf.Session() as sess:
    save_path = "./save100/model.ckpt"
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,save_path)
    label12 = sess.run(label12)
    labely12 = sess.run(labely12)

    testp = sess.run(pr, feed_dict={X: datay, Y: labely12, is_training: False, P: start1})
    testY = np.array(labely)
    testa = sess.run(a, feed_dict={X: datay, Y: labely12, is_training: False, P: start1})
    print(testa)

    import matplotlib.pyplot as plt

    plt.plot(testY + 1)
    plt.plot(testp + 1)
    plt.show()
