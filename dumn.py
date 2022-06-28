#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-06-27

"""
Deep User Match Network for Click-Through Rate Prediction
DUMN Model 模型复现参考
"""
from tensorflow.keras.models import Model

from data_process.feature_process import build_rank_dumn_feature_columns
from layers.model_layers import UserRepresentationLayer, UserMatchLayer, DNNLayer
from layers.tool_layers import *
from tensorflow.keras.optimizers import Adam


def build_dumn_model(city_dict, shangquan_dict, comm_dict, price_dict, area_dict, layer_units):
    # 输入、特征
    feature_columns = build_rank_dumn_feature_columns(city_dict, shangquan_dict, comm_dict, price_dict, area_dict)

    ru = UserRepresentationLayer(name='ru')([feature_columns['em'], feature_columns['eu'], feature_columns['Xu']])

    ru1 = UserRepresentationLayer(name='ru1')([feature_columns['em'], feature_columns['eu1'], feature_columns['Xu1']])

    ru2 = UserRepresentationLayer(name='ru2')([feature_columns['em'], feature_columns['eu2'], feature_columns['Xu2']])

    ru3 = UserRepresentationLayer(name='ru3')([feature_columns['em'], feature_columns['eu3'], feature_columns['Xu3']])

    userMatchLayerOut = UserMatchLayer()([ru, ru1, ru2, ru3])
    Su = userMatchLayerOut['Su']
    Ru = userMatchLayerOut['Ru']

    # CONCAT LAYER
    Em = feature_columns['em']  # candidate item features
    Ec = concatenate([
        feature_columns['continue_features'],
        feature_columns['cross_features']
    ], axis=-1, name='context_features')  # context features

    feature_concat = concatenate([Su, Ru, ru, Em, Ec])

    # output layer
    p = DNNLayer(layer_units=layer_units, dropout_rate=0.3)(feature_concat, True)
    out = Dense(1, activation='sigmoid', name='ctr_predictions')(p)

    model = Model(inputs=feature_columns['total_inputs'], outputs=out)

    optimizer = Adam(1e-5)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])

    model.summary()

    return model
