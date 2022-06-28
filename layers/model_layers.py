#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28
# update: 添加注释，完善自定义层 by stefan 2022-03-02

import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.backend import expand_dims, repeat_elements, sum
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from layers.tool_layers import L2_norm_layer

class DNNLayer(Layer):
    def __init__(self, layer_units, dropout_rate=0.3, **kwargs):
        super(DNNLayer, self).__init__(**kwargs)
        self.layer_units = layer_units
        self.batch_norm = BatchNormalization()
        self.dropout_rate = dropout_rate
        self.dense_layers = []

    def build(self, input_shape):
        super(DNNLayer, self).build(input_shape)
        for i in range(len(self.layer_units)):
            dense_layer = Dense(self.layer_units[i], activation=None)
            self.dense_layers.append(dense_layer)

    def call(self, inputs, training=False):
        net = inputs
        for i in range(len(self.dense_layers)):
            net = self.dense_layers[i](net)
            if i == 0:
                net = self.batch_norm(net)  # batch_norm加在第一层的输入的线性变换后，激活函数前
            net = ReLU()(net)
            if training:
                net = Dropout(self.dropout_rate)(net)
        return net


class UserRepresentationLayer(Layer):
    def __init__(self, **kwargs):
        super(UserRepresentationLayer, self).__init__(**kwargs)
        self.ActivationSumPoolingFromDIN = ActivationSumPoolingFromDIN()

    def call(self, inputs):
        em, eu, Xu = inputs
        ru_ = self.ActivationSumPoolingFromDIN([Xu, em])

        # ru: user representation
        ru = concatenate([ru_, eu], axis=-1)
        return ru


class UserMatchLayer(Layer):
    def __init__(self, **kwargs):
        super(UserMatchLayer, self).__init__(**kwargs)
        self.l2_norm_layer = L2_norm_layer(axis=-1)

    def relavant_unit(self, ru, r_ul):
        ru_norm = self.l2_norm_layer(ru)
        r_ul_norm = self.l2_norm_layer(r_ul)
        a_l = tf.reduce_sum(tf.multiply(ru_norm, r_ul_norm), axis=1, keepdims=True)

        relavant = {'relavant': tf.multiply(a_l, r_ul),
                  'a_l': a_l
                  }
        return relavant

    def call(self, inputs):
        ru, ru1, ru2, ru3 = inputs
        ru_u1 = self.relavant_unit(ru, ru1)
        ru_u2 = self.relavant_unit(ru, ru2)
        ru_u3 = self.relavant_unit(ru, ru3)

        result = {'Su': ru_u1['relavant'] + ru_u2['relavant'] + ru_u3['relavant'],
                  'Ru': ru_u1['a_l'] + ru_u2['a_l'] + ru_u3['a_l']
                  }
        return result



class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='relu'):
        """
            Input shape
                - query: 2D tensor with shape: ``(batch_size, input_dim)``.
                - key: 3D tensor with shape: ``(batch_size, seq_len, input_dim)``.
                - value: 3D tensor with shape: ``(batch_size, seq_len, input_dim)``.
            Output shape
                - 2D tensor with shape: ``(batch_size, input_dim)``.
        """
        super(Attention_Layer, self).__init__()
        self.att_dense = []
        self.att_hidden_units = att_hidden_units
        self.activation = activation
        self.att_final_dense = Dense(1)

    def build(self, input_shape):
        super(Attention_Layer, self).build(input_shape)
        for i in range(len(self.att_hidden_units)):
            self.att_dense.append(Dense(self.att_hidden_units[i], activation=self.activation))

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        q, k, v = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)

        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class ActivationSumPoolingFromDIN(Layer):
    def __init__(self, att_hidden_units=[64, 32], att_activation='relu'):
        """
        用户行为序列对候选集做atten，然后sum pooling
        """
        super(ActivationSumPoolingFromDIN, self).__init__()

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)

    def call(self, inputs):
        seq_embed, item_embed = inputs
        user_interest_sum_pool = self.attention_layer([item_embed, seq_embed, seq_embed])

        # concat user_info(att hist), cadidate item embedding
        info_all = tf.concat([user_interest_sum_pool, item_embed], axis=-1)
        info_all = self.bn(info_all)
        return info_all


class DeepCrossLayer(Layer):
    def __init__(self, layer_num, embed_dim, output_dim=0, **kwargs):
        """
            DCN Model implements
            usage: DeepCrossLayer(2, item_feature.shape[-1], name="deep_cross_features")(item_feature)
        """
        super(DeepCrossLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.embed_dim = embed_dim

        self.w = []
        self.b = []
        for i in range(self.layer_num):
            self.w.append(tf.Variable(lambda: tf.random.truncated_normal(shape=(self.embed_dim,), stddev=0.01)))
            self.b.append(tf.Variable(lambda: tf.zeros(shape=(embed_dim,))))

        self.output_dim = output_dim
        self.dense = Dense(units=self.output_dim, use_bias=False)

    def cross_layer(self, inputs, i):
        x0, xl = inputs
        # feature crossing
        x1_T = tf.reshape(xl, [-1, 1, self.embed_dim])
        x_lw = tf.tensordot(x1_T, self.w[i], axes=1)
        cross = x0 * x_lw
        return cross + self.b[i] + xl

    def call(self, inputs):
        xl = inputs
        for i in range(self.layer_num):
            xl = self.cross_layer([inputs, xl], i)
        if self.output_dim > 0:
            xl = self.dense(xl)
        return xl


