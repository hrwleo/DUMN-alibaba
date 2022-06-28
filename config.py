#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-03-01

import tensorflow as tf
import datetime

"""
909-rank-dumn
模型相关参数配置
"""

flags = tf.compat.v1.flags

flags.DEFINE_string("model_pb", "./model_pb", "Base directory for the item model.")

flags.DEFINE_string("city_dict", "../../demo_data/city_dict", "Path to the city_dict.")
flags.DEFINE_string("shangquan_dict", "../../demo_data/shangquan_dict", "Path to the shangquan_dict.")
flags.DEFINE_string("comm_dict", "../../demo_data/comm_dict", "Path to the comm_dict.")
flags.DEFINE_string("price_dict", "../../demo_data/price_dict", "Path to the price_dict.")
flags.DEFINE_string("area_dict", "../../demo_data/area_dict", "Path to the area_dict.")

flags.DEFINE_string("train_data", "../../demo_data/part-r-00003", "Path to the train data")
flags.DEFINE_string("eval_data", "../../demo_data/part-r-00003", "Path to the evaluation data.")

flags.DEFINE_string("online_logs", "./online_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                    "Path to the log.")

flags.DEFINE_integer("batch_size", 1024, "Training batch size")  # 40960
flags.DEFINE_integer("epoch", 2, "Training epochs")  # 40
flags.DEFINE_string("layer_units", "256,128", "hidden units of layers")

FLAGS = flags.FLAGS