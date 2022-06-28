#!/usr/bin/env python
# coding: utf-8

from common_utils import *
from data_config import *

import config
from dumn import build_dumn_model

FLAGS = config.FLAGS

# read data
train_set = read_data(path=FLAGS.train_data, batch_size=FLAGS.batch_size, if_shuffle=True,
                      feat_desc=data_config["rank-dumn"], if_mtl=False)
test_set = read_data(path=FLAGS.eval_data, batch_size=FLAGS.batch_size, feat_desc=data_config["rank-dumn"], if_mtl=False)

model = build_dumn_model(FLAGS.city_dict, FLAGS.shangquan_dict, FLAGS.comm_dict, FLAGS.price_dict,
                                            FLAGS.area_dict,
                                            FLAGS.layer_units.split(','))

# define callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.online_logs, embeddings_freq=1, update_freq=100,
                                                      embeddings_data=train_set)

model.fit(
    x=train_set,
    epochs=FLAGS.epoch,
    validation_data=test_set,
    callbacks=[tensorboard_callback]
)

model.save(FLAGS.model_pb, save_format='tf', include_optimizer=False)

# 加载模型
dumn_model = tf.keras.models.load_model(FLAGS.model_pb)

print(dumn_model.signatures["serving_default"].inputs)
