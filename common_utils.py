#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28


import tensorflow as tf
import os


def parse_exmp(example_proto, feature_description, if_mtl):
    feature_dict = tf.io.parse_single_example(example_proto, feature_description)
    if if_mtl:
        ctr_label = tf.reshape(feature_dict.pop("is_click"), (1,))
        ctcvr_label = tf.reshape(feature_dict.pop("is_link"), (1,))
        label = tf.concat([ctr_label, ctcvr_label], axis=0)
        return feature_dict, label
    else:
        label = feature_dict.pop("is_click")
        return feature_dict, label


def get_file_list(data):
    if isinstance(data, str) and os.path.isdir(data):
        files = [data + '/' + x for x in os.listdir(data)] if os.path.isdir(data) else data
    else:
        files = data

    return files


def read_data(path, shuffle_buffer_size=20000, batch_size=2048, if_shuffle=False, feat_desc=None, if_mtl=False, if_train=True):
    file_names = get_file_list(path)
    dataset = tf.data.Dataset.list_files(file_names)
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(file_names),
        cycle_length=8
    )
    if if_shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(lambda x: parse_exmp(x, feat_desc, if_mtl), num_parallel_calls=8)

    dataset = dataset.batch(batch_size)
    return dataset


