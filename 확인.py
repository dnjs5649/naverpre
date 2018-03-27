import pickle
from pprint import pprint
import numpy as np
import tensorflow as tf
import time

g_1 = [[[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1]]]

def xavier_initializer(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)
Y = tf.placeholder(tf.float32,[None,5])
X = tf.placeholder(tf.float32,[None,32,128])
Z = tf.placeholder('float')


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

def make_attention(target_embed, encoder_output, decoder_hidden, layer_idx):
    with tf.variable_scope("attention_layer_" + str(layer_idx)):
        embed_size = target_embed.get_shape().as_list()[-1]  # k
        dec_hidden_proj = linear_mapping_weightnorm(decoder_hidden, embed_size,
                                                    var_scope_name="linear_mapping_att_query")  # M*N1*k1 --> M*N1*k
        dec_rep = (dec_hidden_proj + target_embed) * tf.sqrt(0.5)

        encoder_output_a = encoder_output.outputs
        encoder_output_c = encoder_output.attention_values  # M*N2*K

        att_score = tf.matmul(dec_rep, encoder_output_a, transpose_b=True)  # M*N1*K  ** M*N2*K  --> M*N1*N2
        att_score = tf.nn.softmax(att_score)

        length = tf.cast(tf.shape(encoder_output_c), tf.float32)
        att_out = tf.matmul(att_score, encoder_output_c) * length[1] * tf.sqrt(
            1.0 / length[1])  # M*N1*N2  ** M*N2*K   --> M*N1*k

        att_out = linear_mapping_weightnorm(att_out, decoder_hidden.get_shape().as_list()[-1],
                                            var_scope_name="linear_mapping_att_out")
    return att_out



def gated_linear_units(inputs):
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 3
  input_pass = inputs[:,:,0:int(input_shape[2]/2)]
  input_gate = inputs[:,:,int(input_shape[2]/2):]
  input_gate = tf.sigmoid(input_gate)
  return tf.multiply(input_pass, input_gate)

next_layer = tf.get_variable(shape=[1,5,128],name='a')

next_layer = tf.pad(next_layer, [[0, 0], [3 - 1, 3 - 1], [0, 0]],
                            "CONSTANT")
pprint(next_layer)
next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=3, out_dim=128 * 2,
                                       kernel_size=3, padding="VALID",
                                       dropout=0.9, var_scope_name="conv_layer_" + str(3))
pprint(next_layer)

next_layer = next_layer[:, 0:-3 + 1, :]
pprint(next_layer)
next_layer = gated_linear_units(next_layer)
pprint(next_layer)



encoder_output_a = tf.get_variable(shape=[1,3,128],name='b')
encoder_output_c = tf.get_variable(shape=[1,3,128],name='c') # M*N2*K

att_score = tf.matmul(next_layer, encoder_output_a, transpose_b=True)  # M*N1*K  ** M*N2*K  --> M*N1*N2
att_score = tf.nn.softmax(att_score)
pprint(att_score)
length = tf.cast(tf.shape(encoder_output_c), tf.float32)
pprint(length)

att_out = tf.matmul(att_score, encoder_output_c) * length[1] * tf.sqrt(
            1.0 / length[1])
pprint(att_out)
att_out = linear_mapping_weightnorm(att_out, 256,
                                            var_scope_name="linear_mapping_att_out")
pprint(att_out)

next_layer = att_out
pprint(next_layer[:, -1:, :])
next_layer = linear_mapping_weightnorm(next_layer[:, -1:, :], 256,
                                                   var_scope_name="linear_mapping_after_cnn")
pprint(next_layer)

logits = tf.reshape(next_layer, [-1, 256])

pprint(logits)

a=[1,2,3,4,5,6,7,8,9]
pprint(a[-1:])




a=[[[1,0,0,0,0]]]
b=[[[0,1,0,0,0]]]
c=[[[0,0,1,0,0]]]
d=[[[0,0,0,1,0]]]
e=[[[0,0,0,0,1]]]
bv = np.concatenate([a,b],axis=1)
pprint(bv)
pprint(bv.shape)