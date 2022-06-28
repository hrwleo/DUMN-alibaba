#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-03-24

"""
rank 模型特征处理
"""
from layers.model_layers import ActivationSumPoolingFromDIN, DeepCrossLayer
from layers.tool_layers import *


def build_rank_dumn_feature_columns(city_dict, shangquan_dict, comm_dict, price_dict, area_dict):
    # ************************************************************************
    # 特征列定义
    # ************************************************************************
    user_city_seq = tf.keras.Input(shape=(5,), name='user_city_seq', dtype=tf.int64)
    user_shangquan_seq = tf.keras.Input(shape=(5,), name='user_shangquan_seq', dtype=tf.int64)
    user_comm_seq = tf.keras.Input(shape=(5,), name='user_comm_seq', dtype=tf.int64)
    user_price_seq = tf.keras.Input(shape=(5,), name='user_price_seq', dtype=tf.int64)
    user_area_seq = tf.keras.Input(shape=(5,), name='user_area_seq', dtype=tf.int64)

    item_user1_city_seq = tf.keras.Input(shape=(5,), name='item_user1_city_seq', dtype=tf.int64)
    item_user1_shangquan_seq = tf.keras.Input(shape=(5,), name='item_user1_shangquan_seq', dtype=tf.int64)
    item_user1_comm_seq = tf.keras.Input(shape=(5,), name='item_user1_comm_seq', dtype=tf.int64)
    item_user1_price_seq = tf.keras.Input(shape=(5,), name='item_user1_price_seq', dtype=tf.int64)
    item_user1_area_seq = tf.keras.Input(shape=(5,), name='item_user1_area_seq', dtype=tf.int64)

    ...

    # common emb 区域类特征在底层共享
    city_Embedding = Embedding(input_dim=400, output_dim=16, mask_zero=False, name="city_emb")
    shangquan_Embedding = Embedding(input_dim=15000, output_dim=32, mask_zero=False, name="shangquan_emb")
    comm_Embedding = Embedding(input_dim=400000, output_dim=32, mask_zero=False, name="comm_emb")
    price_Embedding = Embedding(input_dim=50, output_dim=4, mask_zero=False, name="price_emb")
    area_Embedding = Embedding(input_dim=50, output_dim=4, mask_zero=False, name="area_emb")

    # ************************************************************************
    # 特征分类： 序列embedding, 离散embedding
    # ************************************************************************

    # 1. embedding ***********************************************************
    user_city_id_token = VocabLayer(city_dict, 'city_token')(user_city_seq)
    user_city_emb_seq = city_Embedding(user_city_id_token)  # 以city_id为index取emb  shape(None, 5, emb_size)
    user_city_emb = GlobalAveragePooling1D()(user_city_emb_seq)  # shape(None, emb_size)

    ...

    # em: candidate item embedding
    em = concatenate([item_city_emb, item_shangquan_emb, item_comm_emb, item_price_emb, item_area_emb],
                     axis=-1, name='em')

    # eu: target_user embedding
    eu = concatenate([user_city_emb, user_shangquan_emb, user_comm_emb, user_price_emb, user_area_emb],
                     axis=-1, name='eu')

    # Xu target_user history embedding
    Xu = getXulEmbedding(user_city_emb_seq, user_shangquan_emb_seq, user_comm_emb_seq, user_price_emb_seq,
                         user_area_emb_seq, 'Xu')

    # # item -> target_user history attention
    # ru_ = ActivationSumPoolingFromDIN()([Xu, em])
    #
    # # r_u: target user representation
    # ru = concatenate([ru_, eu], axis=-1, name='ru')

    eu1 = concatenate([...], name='eu1')

    Xu1 = getXulEmbedding(..., 'Xu1')

    eu2 = 

    Xu2 = 

    eu3 = 

    Xu3 = 
    # context feature
    # cross tower
    category_features = concatenate([...], axis=1, name='category_features')

    dcn_features = DeepCrossLayer(2, category_features.shape[-1], name='dcn_features')(category_features)

    # 2.连续特征 ************************************************************************
    continue_inputs = [...]

    continue_features = concatenate([...], axis=1, name='continue_features')

    total_inputs = embedding_inputs + continue_inputs

    result = {'total_inputs': total_inputs,
              'continue_features': continue_features,
              'cross_features': dcn_features,
              'em': em, 'eu': eu, 'Xu': Xu, 'eu1': eu1, 'Xu1': Xu1, 'eu2': eu2, 'Xu2': Xu2, 'eu3': eu3, 'Xu3': Xu3
              }

    return result


def getXulEmbedding(user_city_emb_seq, user_shangquan_emb_seq, user_comm_emb_seq, user_price_emb_seq,
                    user_area_emb_seq, tag):
    Xu = []
    for i in range(5):
        city_emb = tf.slice(user_city_emb_seq, [0, i, 0], [-1, 1, -1])
        shangquan_emb = tf.slice(user_shangquan_emb_seq, [0, i, 0], [-1, 1, -1])
        comm_emb = tf.slice(user_comm_emb_seq, [0, i, 0], [-1, 1, -1])
        price_emb = tf.slice(user_price_emb_seq, [0, i, 0], [-1, 1, -1])
        area_emb = tf.slice(user_area_emb_seq, [0, i, 0], [-1, 1, -1])
        item_emb = concatenate([city_emb, shangquan_emb, comm_emb, price_emb, area_emb], axis=-1)
        Xu.append(item_emb)

    Xu = concatenate(Xu, axis=-2, name=tag + '_emb') 
    return Xu
