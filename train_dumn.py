#!/usr/bin/env python
# coding: utf-8

# import sys
# sys.path.insert(0, r'/code/Stefan/909_rank/dm-recommend-tf2/')  # 线上要加入搜索目录的路径
from data_process.common_utils import *
from data_process.data_config import *

from rank.DUMN import config
from rank.DUMN.dumn import build_dumn_model

FLAGS = config.FLAGS

# read data
train_set = read_data(path=FLAGS.train_data, batch_size=FLAGS.batch_size, if_shuffle=True,
                      feat_desc=data_config["909-rank-dumn"], if_mtl=False)
test_set = read_data(path=FLAGS.eval_data, batch_size=FLAGS.batch_size, feat_desc=data_config["909-rank-dumn"], if_mtl=False)

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

# two optimizer in wide&deep can not be serialized, excluding optimizer is ok for prediction
model.save(FLAGS.model_pb, save_format='tf', include_optimizer=False)

# 加载模型
dumn_model = tf.keras.models.load_model(FLAGS.model_pb)

print(dumn_model.signatures["serving_default"].inputs)
