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

def gated_linear_units(inputs):
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 3
  input_pass = inputs[:,:,0:int(input_shape[2]/2)]
  input_gate = inputs[:,:,int(input_shape[2]/2):]
  input_gate = tf.sigmoid(input_gate)
  return tf.multiply(input_pass, input_gate)


def conv_decoder_stack(target_embed, enc_output, inputs, nhids_list, kwidths_list, dropout_dict, mode):
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
        layer_shape = next_layer.get_shape().as_list()
        assert len(layer_shape) == 3
        # to avoid using future information
        next_layer = next_layer[:, 0:-kwidths_list[layer_idx] + 1, :]

        next_layer = gated_linear_units(next_layer)

        # add attention
        # decoder output -->linear mapping to embed, + target embed,  query decoder output a, softmax --> scores, scores*encoder_output_c-->output,  output--> linear mapping to nhid+  decoder_output -->
        att_out = make_attention(target_embed, enc_output, next_layer, layer_idx)
        next_layer = (next_layer + att_out) * tf.sqrt(0.5)

        # add res connections
        next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
    return next_layer



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


def step(self, time, inputs, state, name=None):
    cur_inputs = inputs[:, 0:time + 1, :]
    zeros_padding = inputs[:, time + 2:, :]
    cur_inputs_pos = self.add_position_embedding(cur_inputs, time)

    enc_output = state
    logits = self.infer_conv_block(enc_output, cur_inputs_pos)

    sample_ids = tf.cast(tf.argmax(logits, axis=-1), dtypes.int32)

    finished, next_inputs = self.next_inputs(sample_ids=sample_ids)
    next_inputs = tf.reshape(next_inputs, [self.config.beam_width, 1, inputs.get_shape().as_list()[-1]])
    next_inputs = tf.concat([cur_inputs, next_inputs], axis=1)
    next_inputs = tf.concat([next_inputs, zeros_padding], axis=1)
    next_inputs.set_shape([self.config.beam_width, self.params['max_decode_length'], inputs.get_shape().as_list()[-1]])
    outputs = ConvDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids)
    return outputs, enc_output, next_inputs, finished


def infer_conv_block(self, enc_output, input_embed):
    # Apply dropout to embeddings
    input_embed = tf.contrib.layers.dropout(
        inputs=input_embed,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.INFER)

    next_layer = self.conv_block(enc_output, input_embed, False)
    shape = next_layer.get_shape().as_list()

    logits = tf.reshape(next_layer, [-1, shape[-1]])
    return logits


def conv_block(self, enc_output, input_embed, is_train=True):
    with tf.variable_scope("decoder_cnn"):
        next_layer = input_embed
        if self.params["cnn.layers"] > 0:
            nhids_list = parse_list_or_default(self.params["cnn.nhids"], self.params["cnn.layers"],
                                               self.params["cnn.nhid_default"])
            kwidths_list = parse_list_or_default(self.params["cnn.kwidths"], self.params["cnn.layers"],
                                                 self.params["cnn.kwidth_default"])

            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0],
                                                   dropout=self.params["embedding_dropout_keep_prob"],
                                                   var_scope_name="linear_mapping_before_cnn")

            next_layer = conv_decoder_stack(input_embed, enc_output, next_layer, nhids_list, kwidths_list,
                                            {'src': self.params["embedding_dropout_keep_prob"],
                                             'hid': self.params["nhid_dropout_keep_prob"]}, mode=self.mode)

    with tf.variable_scope("softmax"):
        if is_train:
            next_layer = linear_mapping_weightnorm(next_layer, self.params["nout_embed"],
                                                   var_scope_name="linear_mapping_after_cnn")
        else:
            next_layer = linear_mapping_weightnorm(next_layer[:, -1:, :], self.params["nout_embed"],
                                                   var_scope_name="linear_mapping_after_cnn")
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=self.params["out_dropout_keep_prob"],
            is_training=is_train)

        next_layer = linear_mapping_weightnorm(next_layer, self.vocab_size, in_dim=self.params["nout_embed"],
                                               dropout=self.params["out_dropout_keep_prob"],
                                               var_scope_name="logits_before_softmax")

    return next_layer






deco = conv_block()