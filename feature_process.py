#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-03-24

"""
rank 模型特征处理
"""
from layers.model_layers import ActivationSumPoolingFromDIN, DeepCrossLayer
from layers.tool_layers import *


def build_rank_feature_columns(city_dict, shangquan_dict, comm_dict, price_dict, area_dict):
    user_city_seq = tf.keras.Input(shape=(5,), name='user_city_seq', dtype=tf.int64)
    user_shangquan_seq = tf.keras.Input(shape=(5,), name='user_shangquan_seq', dtype=tf.int64)
    user_comm_seq = tf.keras.Input(shape=(5,), name='user_comm_seq', dtype=tf.int64)
    user_price_seq = tf.keras.Input(shape=(5,), name='user_price_seq', dtype=tf.int64)
    user_area_seq = tf.keras.Input(shape=(5,), name='user_area_seq', dtype=tf.int64)

    item_city_id = tf.keras.Input(shape=(1,), name='city_id', dtype=tf.int64)
    item_comm_id = tf.keras.Input(shape=(1,), name='comm_id', dtype=tf.int64)
    item_shangquan_id = tf.keras.Input(shape=(1,), name='shangquan_id', dtype=tf.int64)
    item_price_id = tf.keras.Input(shape=(1,), name='price_id', dtype=tf.int64)
    item_area_id = tf.keras.Input(shape=(1,), name='area_id', dtype=tf.int64)
    item_floor_loc = tf.keras.Input(shape=(1,), name='floor_loc', dtype=tf.int64)
    item_total_floor = tf.keras.Input(shape=(1,), name='total_floor', dtype=tf.int64)
    item_room_num = tf.keras.Input(shape=(1,), name='room_num', dtype=tf.int64)
    item_hall = tf.keras.Input(shape=(1,), name='hall', dtype=tf.int64)
    item_bathroom = tf.keras.Input(shape=(1,), name='bathroom', dtype=tf.int64)

    item_pqs = tf.keras.Input(shape=(1,), name='pqs', dtype=tf.float32)
    item_prop_age = tf.keras.Input(shape=(1,), name='prop_age', dtype=tf.float32)
    item_edu_link_rate = tf.keras.Input(shape=(1,), name='edu_link_rate', dtype=tf.float32)
    item_floor_link_rate = tf.keras.Input(shape=(1,), name='floor_link_rate', dtype=tf.float32)

    item_orient = tf.keras.Input(shape=(1,), name='orient', dtype=tf.int64)
    item_fitment = tf.keras.Input(shape=(1,), name='fitment', dtype=tf.int64)
    item_is_guarantee = tf.keras.Input(shape=(1,), name='is_guarantee', dtype=tf.int64)
    item_is_media = tf.keras.Input(shape=(1,), name='is_media', dtype=tf.int64)
    item_is_720 = tf.keras.Input(shape=(1,), name='is_720', dtype=tf.int64)

    item_green_rate = tf.keras.Input(shape=(1,), name='green_rate', dtype=tf.float32)
    item_traffic = tf.keras.Input(shape=(1,), name='traffic', dtype=tf.float32)
    item_education = tf.keras.Input(shape=(1,), name='education', dtype=tf.float32)
    item_business = tf.keras.Input(shape=(1,), name='business', dtype=tf.float32)
    item_environment = tf.keras.Input(shape=(1,), name='environment', dtype=tf.float32)
    item_popularity = tf.keras.Input(shape=(1,), name='popularity', dtype=tf.float32)
    item_impression_score = tf.keras.Input(shape=(1,), name='impression_score', dtype=tf.float32)
    item_comm_score = tf.keras.Input(shape=(1,), name='comm_score', dtype=tf.float32)

    # common emb 区域类特征在底层共享
    city_Embedding = Embedding(input_dim=400, output_dim=16, mask_zero=False, name="city_emb")
    shangquan_Embedding = Embedding(input_dim=15000, output_dim=32, mask_zero=False, name="shangquan_emb")
    comm_Embedding = Embedding(input_dim=400000, output_dim=32, mask_zero=False, name="comm_emb")
    price_Embedding = Embedding(input_dim=50, output_dim=4, mask_zero=False, name="price_emb")
    area_Embedding = Embedding(input_dim=50, output_dim=4, mask_zero=False, name="area_emb")

    # ************************************************************************
    # 特征分类： 序列embedding, 离散embedding
    # ************************************************************************

    # 1. embedding ************************************************************************
    user_city_id_token = VocabLayer(city_dict, 'city_token')(user_city_seq)
    user_city_emb_seq = city_Embedding(user_city_id_token)  # 以city_id为index取emb  shape(None, 5, emb_size)
    user_city_emb = GlobalMaxPooling1D()(user_city_emb_seq)  # shape(None, emb_size)

    user_shangquan_id_token = VocabLayer(shangquan_dict, 'shangquan_token')(user_shangquan_seq)
    user_shangquan_emb_seq = shangquan_Embedding(user_shangquan_id_token)
    user_shangquan_emb = GlobalMaxPooling1D()(user_shangquan_emb_seq)

    user_comm_id_token = VocabLayer(comm_dict, 'comm_token')(user_comm_seq)
    user_comm_emb_seq = comm_Embedding(user_comm_id_token)
    user_comm_emb = GlobalMaxPooling1D()(user_comm_emb_seq)

    user_price_id_token = VocabLayer(price_dict, 'user_price_id_token')(user_price_seq)
    user_price_emb_seq = price_Embedding(user_price_id_token)
    user_price_emb = GlobalMaxPooling1D()(user_price_emb_seq)

    user_area_id_token = VocabLayer(area_dict, 'user_area_id_token')(user_area_seq)
    user_area_emb_seq = area_Embedding(user_area_id_token)
    user_area_emb = GlobalMaxPooling1D()(user_area_emb_seq)

    item_city_id_token = VocabLayer(city_dict, 'item_city_token')(item_city_id)
    item_city_emb = city_Embedding(item_city_id_token)
    item_city_emb = Reshape((16,))(item_city_emb)

    item_shangquan_id_token = VocabLayer(shangquan_dict, 'item_shangquan_token')(item_shangquan_id)
    item_shangquan_emb = shangquan_Embedding(item_shangquan_id_token)
    item_shangquan_emb = Reshape((32,))(item_shangquan_emb)

    item_comm_id_token = VocabLayer(comm_dict, 'item_comm_token')(item_comm_id)
    item_comm_emb = comm_Embedding(item_comm_id_token)
    item_comm_emb = Reshape((32,))(item_comm_emb)

    item_floor_emb = HashBucketsEmbedding(50, 4, name='item_floor_emb')(item_floor_loc)
    item_floor_emb = Reshape((4,))(item_floor_emb)

    item_total_floor_emb = HashBucketsEmbedding(50, 4, name='item_total_floor_emb')(item_total_floor)
    item_total_floor_emb = Reshape((4,))(item_total_floor_emb)

    item_price_id_token = VocabLayer(price_dict, 'item_price_id_token')(item_price_id)
    item_price_emb = price_Embedding(item_price_id_token)
    item_price_emb = Reshape((4,))(item_price_emb)

    item_area_id_token = VocabLayer(area_dict, 'item_area_id_token')(item_area_id)
    item_area_emb = area_Embedding(item_area_id_token)
    item_area_emb = Reshape((4,))(item_area_emb)

    item_orient_emb = HashBucketsEmbedding(20, 4, name='item_orient_emb')(item_orient)
    item_orient_emb = Reshape((4,))(item_orient_emb)

    item_fitment_emb = HashBucketsEmbedding(20, 4, name='item_fitment_emb')(item_fitment)
    item_fitment_emb = Reshape((4,))(item_fitment_emb)

    item_room_emb = HashBucketsEmbedding(10, 4, name='item_room_emb')(item_room_num)
    item_room_emb = Reshape((4,))(item_room_emb)

    item_hall_emb = HashBucketsEmbedding(10, 4, name='item_hall_emb')(item_hall)
    item_hall_emb = Reshape((4,))(item_hall_emb)

    item_bathroom_emb = HashBucketsEmbedding(10, 4, name='item_bathroom_emb')(item_bathroom)
    item_bathroom_emb = Reshape((4,))(item_bathroom_emb)

    item_is_guarantee_emb = HashBucketsEmbedding(2, 2, name='item_is_guarantee_emb')(item_is_guarantee)
    item_is_guarantee_emb = Reshape((2,))(item_is_guarantee_emb)

    item_is_media_emb = HashBucketsEmbedding(2, 2, name='item_is_media_emb')(item_is_media)
    item_is_media_emb = Reshape((2,))(item_is_media_emb)

    item_is_720_emb = HashBucketsEmbedding(2, 2, name='item_is_720_emb')(item_is_720)
    item_is_720_emb = Reshape((2,))(item_is_720_emb)

    embedding_inputs = [user_city_seq, user_shangquan_seq, user_comm_seq, user_price_seq, user_area_seq,
                        item_city_id, item_shangquan_id, item_comm_id, item_floor_loc, item_total_floor,
                        item_orient, item_fitment, item_price_id, item_area_id, item_room_num, item_hall,
                        item_bathroom, item_is_guarantee, item_is_media, item_is_720]

    # Max-pooling and Din attention Tower
    candidate_emb = tf.concat([item_city_emb, item_shangquan_emb, item_comm_emb, item_price_emb, user_area_emb],
                              axis=-1)

    user_seq_emb = []
    for i in range(5):
        city_emb = tf.slice(user_city_emb_seq, [0, i, 0], [-1, 1, -1])
        shangquan_emb = tf.slice(user_shangquan_emb_seq, [0, i, 0], [-1, 1, -1])
        comm_emb = tf.slice(user_comm_emb_seq, [0, i, 0], [-1, 1, -1])
        price_emb = tf.slice(user_price_emb_seq, [0, i, 0], [-1, 1, -1])
        area_emb = tf.slice(user_area_emb_seq, [0, i, 0], [-1, 1, -1])
        user_emb = tf.concat([city_emb, shangquan_emb, comm_emb, price_emb, area_emb], axis=-1)
        user_seq_emb.append(user_emb)

    user_seq_emb = tf.concat(user_seq_emb, axis=-2)  # 注意这里要以第2维拼接 （None, 5, emb_size）
    din_attention_out_emb = ActivationSumPoolingFromDIN()([user_seq_emb, candidate_emb])

    din_max_pooling_out_emb = concatenate([user_city_emb, user_shangquan_emb, user_comm_emb, user_price_emb,
                                           user_area_emb, din_attention_out_emb], axis=-1,
                                          name='din_max_pooling_out_emb')

    # cross tower
    category_features = concatenate([item_floor_emb, item_total_floor_emb, item_orient_emb, item_fitment_emb,
                                     item_room_emb, item_hall_emb, item_bathroom_emb, item_is_guarantee_emb,
                                     item_is_media_emb, item_is_720_emb], axis=1, name='category_features')

    dcn_features = DeepCrossLayer(2, category_features.shape[-1], name='cross_tower')(category_features)

    # all emb
    embedding_features = concatenate([user_city_emb, user_shangquan_emb, user_comm_emb, user_price_emb,
                                      item_city_emb, user_area_emb, item_shangquan_emb, item_comm_emb,
                                      item_floor_emb, item_total_floor_emb, item_price_emb, item_area_emb,
                                      item_orient_emb, item_fitment_emb, item_room_emb, item_hall_emb,
                                      item_bathroom_emb, item_is_guarantee_emb, item_is_media_emb,
                                      item_is_720_emb],
                                     axis=1, name='embedding_features')

    # 2.连续特征 ************************************************************************
    continue_inputs = [item_pqs, item_prop_age, item_edu_link_rate, item_floor_link_rate, item_green_rate, item_traffic,
                       item_education, item_business, item_environment, item_popularity, item_impression_score,
                       item_comm_score]

    continue_features = concatenate([item_pqs, item_prop_age, item_edu_link_rate, item_floor_link_rate, item_green_rate,
                                     item_traffic, item_education, item_business, item_environment, item_popularity,
                                     item_impression_score, item_comm_score], axis=1, name='continue_features')

    total_inputs = embedding_inputs + continue_inputs
    total_features = concatenate([embedding_features, continue_features], axis=1, name='total_features')

    result = {'total_inputs': total_inputs,
              'continue_features': continue_features,
              'category_features': category_features,
              'seq_features': din_max_pooling_out_emb,  # seq tower out
              'cross_features': dcn_features,  # cross tower out
              'total_features': total_features
              }

    return result


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

    item_user2_city_seq = tf.keras.Input(shape=(5,), name='item_user2_city_seq', dtype=tf.int64)
    item_user2_shangquan_seq = tf.keras.Input(shape=(5,), name='item_user2_shangquan_seq', dtype=tf.int64)
    item_user2_comm_seq = tf.keras.Input(shape=(5,), name='item_user2_comm_seq', dtype=tf.int64)
    item_user2_price_seq = tf.keras.Input(shape=(5,), name='item_user2_price_seq', dtype=tf.int64)
    item_user2_area_seq = tf.keras.Input(shape=(5,), name='item_user2_area_seq', dtype=tf.int64)

    item_user3_city_seq = tf.keras.Input(shape=(5,), name='item_user3_city_seq', dtype=tf.int64)
    item_user3_shangquan_seq = tf.keras.Input(shape=(5,), name='item_user3_shangquan_seq', dtype=tf.int64)
    item_user3_comm_seq = tf.keras.Input(shape=(5,), name='item_user3_comm_seq', dtype=tf.int64)
    item_user3_price_seq = tf.keras.Input(shape=(5,), name='item_user3_price_seq', dtype=tf.int64)
    item_user3_area_seq = tf.keras.Input(shape=(5,), name='item_user3_area_seq', dtype=tf.int64)

    item_user_city_seq = [item_user1_city_seq, item_user2_city_seq, item_user3_city_seq]
    item_user_shangquan_seq = [item_user1_shangquan_seq, item_user2_shangquan_seq, item_user3_shangquan_seq]
    item_user_comm_seq = [item_user1_comm_seq, item_user2_comm_seq, item_user3_comm_seq]
    item_user_price_seq = [item_user1_price_seq, item_user2_price_seq, item_user3_price_seq]
    item_user_area_seq = [item_user1_area_seq, item_user2_area_seq, item_user3_area_seq]

    item_city_id = tf.keras.Input(shape=(1,), name='city_id', dtype=tf.int64)
    item_comm_id = tf.keras.Input(shape=(1,), name='comm_id', dtype=tf.int64)
    item_shangquan_id = tf.keras.Input(shape=(1,), name='shangquan_id', dtype=tf.int64)
    item_price_id = tf.keras.Input(shape=(1,), name='price_id', dtype=tf.int64)
    item_area_id = tf.keras.Input(shape=(1,), name='area_id', dtype=tf.int64)
    item_floor_loc = tf.keras.Input(shape=(1,), name='floor_loc', dtype=tf.int64)
    item_total_floor = tf.keras.Input(shape=(1,), name='total_floor', dtype=tf.int64)
    item_room_num = tf.keras.Input(shape=(1,), name='room_num', dtype=tf.int64)
    item_hall = tf.keras.Input(shape=(1,), name='hall', dtype=tf.int64)
    item_bathroom = tf.keras.Input(shape=(1,), name='bathroom', dtype=tf.int64)

    item_pqs = tf.keras.Input(shape=(1,), name='pqs', dtype=tf.float32)
    item_prop_age = tf.keras.Input(shape=(1,), name='prop_age', dtype=tf.float32)
    item_edu_link_rate = tf.keras.Input(shape=(1,), name='edu_link_rate', dtype=tf.float32)
    item_floor_link_rate = tf.keras.Input(shape=(1,), name='floor_link_rate', dtype=tf.float32)

    item_orient = tf.keras.Input(shape=(1,), name='orient', dtype=tf.int64)
    item_fitment = tf.keras.Input(shape=(1,), name='fitment', dtype=tf.int64)
    item_is_guarantee = tf.keras.Input(shape=(1,), name='is_guarantee', dtype=tf.int64)
    item_is_media = tf.keras.Input(shape=(1,), name='is_media', dtype=tf.int64)
    item_is_720 = tf.keras.Input(shape=(1,), name='is_720', dtype=tf.int64)

    item_green_rate = tf.keras.Input(shape=(1,), name='green_rate', dtype=tf.float32)
    item_traffic = tf.keras.Input(shape=(1,), name='traffic', dtype=tf.float32)
    item_education = tf.keras.Input(shape=(1,), name='education', dtype=tf.float32)
    item_business = tf.keras.Input(shape=(1,), name='business', dtype=tf.float32)
    item_environment = tf.keras.Input(shape=(1,), name='environment', dtype=tf.float32)
    item_popularity = tf.keras.Input(shape=(1,), name='popularity', dtype=tf.float32)
    item_impression_score = tf.keras.Input(shape=(1,), name='impression_score', dtype=tf.float32)
    item_comm_score = tf.keras.Input(shape=(1,), name='comm_score', dtype=tf.float32)

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

    user_shangquan_id_token = VocabLayer(shangquan_dict, 'shangquan_token')(user_shangquan_seq)
    user_shangquan_emb_seq = shangquan_Embedding(user_shangquan_id_token)
    user_shangquan_emb = GlobalAveragePooling1D()(user_shangquan_emb_seq)

    user_comm_id_token = VocabLayer(comm_dict, 'comm_token')(user_comm_seq)
    user_comm_emb_seq = comm_Embedding(user_comm_id_token)
    user_comm_emb = GlobalAveragePooling1D()(user_comm_emb_seq)

    user_price_id_token = VocabLayer(price_dict, 'user_price_id_token')(user_price_seq)
    user_price_emb_seq = price_Embedding(user_price_id_token)
    user_price_emb = GlobalAveragePooling1D()(user_price_emb_seq)

    user_area_id_token = VocabLayer(area_dict, 'user_area_id_token')(user_area_seq)
    user_area_emb_seq = area_Embedding(user_area_id_token)
    user_area_emb = GlobalAveragePooling1D()(user_area_emb_seq)

    item_user_city_emb_seq = []
    item_user_shangquan_emb_seq = []
    item_user_comm_emb_seq = []
    item_user_price_emb_seq = []
    item_user_area_emb_seq = []

    item_user_city_emb_seq_seq = []
    item_user_shangquan_emb_seq_seq = []
    item_user_comm_emb_seq_seq = []
    item_user_price_emb_seq_seq = []
    item_user_area_emb_seq_seq = []

    for i in range(3):
        city_id_token = VocabLayer(city_dict, 'item_user' + str(i) + '_city_token')(item_user_city_seq[i])
        city_emb_seq = city_Embedding(city_id_token)
        city_emb = GlobalAveragePooling1D(name='item_user' + str(i) + '_city_emb')(
            city_emb_seq)  # shape(None, emb_size)
        item_user_city_emb_seq.append(city_emb)
        item_user_city_emb_seq_seq.append(city_emb_seq)

        shangquan_id_token = VocabLayer(shangquan_dict, 'item_user' + str(i) + '_shangquan_token')(
            item_user_shangquan_seq[i])
        shangquan_emb_seq = shangquan_Embedding(shangquan_id_token)
        shangquan_emb = GlobalAveragePooling1D(name='item_user' + str(i) + '_shangquan_emb')(shangquan_emb_seq)
        item_user_shangquan_emb_seq.append(shangquan_emb)
        item_user_shangquan_emb_seq_seq.append(shangquan_emb_seq)

        comm_id_token = VocabLayer(comm_dict, 'item_user' + str(i) + '_comm_token')(item_user_comm_seq[i])
        comm_emb_seq = comm_Embedding(comm_id_token)
        comm_emb = GlobalAveragePooling1D(name='item_user' + str(i) + '_comm_emb')(comm_emb_seq)
        item_user_comm_emb_seq.append(comm_emb)
        item_user_comm_emb_seq_seq.append(comm_emb_seq)

        price_id_token = VocabLayer(price_dict, 'item_user' + str(i) + '_price_token')(item_user_price_seq[i])
        price_emb_seq = price_Embedding(price_id_token)
        price_emb = GlobalAveragePooling1D(name='item_user' + str(i) + '_price_emb')(price_emb_seq)
        item_user_price_emb_seq.append(price_emb)
        item_user_price_emb_seq_seq.append(price_emb_seq)

        area_id_token = VocabLayer(area_dict, 'item_user' + str(i) + '_area_token')(item_user_area_seq[i])
        area_emb_seq = area_Embedding(area_id_token)
        area_emb = GlobalAveragePooling1D(name='item_user' + str(i) + '_area_emb')(area_emb_seq)
        item_user_area_emb_seq.append(area_emb)
        item_user_area_emb_seq_seq.append(area_emb_seq)

    item_city_id_token = VocabLayer(city_dict, 'item_city_token')(item_city_id)
    item_city_emb = city_Embedding(item_city_id_token)
    item_city_emb = Reshape((16,))(item_city_emb)

    item_shangquan_id_token = VocabLayer(shangquan_dict, 'item_shangquan_token')(item_shangquan_id)
    item_shangquan_emb = shangquan_Embedding(item_shangquan_id_token)
    item_shangquan_emb = Reshape((32,))(item_shangquan_emb)

    item_comm_id_token = VocabLayer(comm_dict, 'item_comm_token')(item_comm_id)
    item_comm_emb = comm_Embedding(item_comm_id_token)
    item_comm_emb = Reshape((32,))(item_comm_emb)

    item_floor_emb = HashBucketsEmbedding(50, 4, name='item_floor_emb')(item_floor_loc)
    item_floor_emb = Reshape((4,))(item_floor_emb)

    item_total_floor_emb = HashBucketsEmbedding(50, 4, name='item_total_floor_emb')(item_total_floor)
    item_total_floor_emb = Reshape((4,))(item_total_floor_emb)

    item_price_id_token = VocabLayer(price_dict, 'item_price_id_token')(item_price_id)
    item_price_emb = price_Embedding(item_price_id_token)
    item_price_emb = Reshape((4,))(item_price_emb)

    item_area_id_token = VocabLayer(area_dict, 'item_area_id_token')(item_area_id)
    item_area_emb = area_Embedding(item_area_id_token)
    item_area_emb = Reshape((4,))(item_area_emb)

    item_orient_emb = HashBucketsEmbedding(20, 4, name='item_orient_emb')(item_orient)
    item_orient_emb = Reshape((4,))(item_orient_emb)

    item_fitment_emb = HashBucketsEmbedding(20, 4, name='item_fitment_emb')(item_fitment)
    item_fitment_emb = Reshape((4,))(item_fitment_emb)

    item_room_emb = HashBucketsEmbedding(10, 4, name='item_room_emb')(item_room_num)
    item_room_emb = Reshape((4,))(item_room_emb)

    item_hall_emb = HashBucketsEmbedding(10, 4, name='item_hall_emb')(item_hall)
    item_hall_emb = Reshape((4,))(item_hall_emb)

    item_bathroom_emb = HashBucketsEmbedding(10, 4, name='item_bathroom_emb')(item_bathroom)
    item_bathroom_emb = Reshape((4,))(item_bathroom_emb)

    item_is_guarantee_emb = HashBucketsEmbedding(2, 2, name='item_is_guarantee_emb')(item_is_guarantee)
    item_is_guarantee_emb = Reshape((2,))(item_is_guarantee_emb)

    item_is_media_emb = HashBucketsEmbedding(2, 2, name='item_is_media_emb')(item_is_media)
    item_is_media_emb = Reshape((2,))(item_is_media_emb)

    item_is_720_emb = HashBucketsEmbedding(2, 2, name='item_is_720_emb')(item_is_720)
    item_is_720_emb = Reshape((2,))(item_is_720_emb)

    embedding_inputs = [user_city_seq, user_shangquan_seq, user_comm_seq, user_price_seq, user_area_seq,
                        item_city_id, item_shangquan_id, item_comm_id, item_floor_loc, item_total_floor,
                        item_orient, item_fitment, item_price_id, item_area_id, item_room_num, item_hall,
                        item_bathroom, item_is_guarantee, item_is_media, item_is_720] + item_user_city_seq \
                       + item_user_shangquan_seq + item_user_comm_seq + item_user_price_seq + item_user_area_seq

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

    eu1 = concatenate([item_user_city_emb_seq[0], item_user_shangquan_emb_seq[0], item_user_comm_emb_seq[0],
                       item_user_price_emb_seq[0],
                       item_user_area_emb_seq[0]], name='eu1')

    Xu1 = getXulEmbedding(item_user_city_emb_seq_seq[0], item_user_shangquan_emb_seq_seq[0],
                          item_user_comm_emb_seq_seq[0],
                          item_user_price_emb_seq_seq[0], item_user_area_emb_seq_seq[0], 'Xu1')

    eu2 = concatenate([item_user_city_emb_seq[1], item_user_shangquan_emb_seq[1], item_user_comm_emb_seq[1],
                       item_user_price_emb_seq[1],
                       item_user_area_emb_seq[1]], name='eu2')

    Xu2 = getXulEmbedding(item_user_city_emb_seq_seq[1], item_user_shangquan_emb_seq_seq[1],
                          item_user_comm_emb_seq_seq[1],
                          item_user_price_emb_seq_seq[1], item_user_area_emb_seq_seq[1], 'Xu2')

    eu3 = concatenate([item_user_city_emb_seq[2], item_user_shangquan_emb_seq[2], item_user_comm_emb_seq[2],
                       item_user_price_emb_seq[2],
                       item_user_area_emb_seq[2]], name='eu3')

    Xu3 = getXulEmbedding(item_user_city_emb_seq_seq[2], item_user_shangquan_emb_seq_seq[2],
                          item_user_comm_emb_seq_seq[2],
                          item_user_price_emb_seq_seq[2], item_user_area_emb_seq_seq[2], 'Xu3')

    # context feature
    # cross tower
    category_features = concatenate([item_floor_emb, item_total_floor_emb, item_orient_emb, item_fitment_emb,
                                     item_room_emb, item_hall_emb, item_bathroom_emb, item_is_guarantee_emb,
                                     item_is_media_emb, item_is_720_emb], axis=1, name='category_features')

    dcn_features = DeepCrossLayer(2, category_features.shape[-1], name='dcn_features')(category_features)

    # 2.连续特征 ************************************************************************
    continue_inputs = [item_pqs, item_prop_age, item_edu_link_rate, item_floor_link_rate, item_green_rate, item_traffic,
                       item_education, item_business, item_environment, item_popularity, item_impression_score,
                       item_comm_score]

    continue_features = concatenate([item_pqs, item_prop_age, item_edu_link_rate, item_floor_link_rate, item_green_rate,
                                     item_traffic, item_education, item_business, item_environment, item_popularity,
                                     item_impression_score, item_comm_score], axis=1, name='continue_features')

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

    Xu = concatenate(Xu, axis=-2, name=tag + '_emb')  # 注意这里要以第2维拼接 （None, 5, emb_size）
    return Xu
