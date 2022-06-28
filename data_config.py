#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28

import tensorflow as tf

rank_config = {
    # user


    # prop


    # label
    "is_click": tf.io.FixedLenFeature([], tf.int64),
    #"is_link": tf.io.FixedLenFeature([], tf.int64),

}


rank_dumn_config = {
    # target user click history
    

    # item-clicked 3 users -> click history


    # prop
    

    # label
    "is_click": tf.io.FixedLenFeature([], tf.int64)
}


recall_config = {
    # user
    
    # prop

    # label
    "is_click": tf.io.FixedLenFeature([], tf.int64),

}

user_emb_909_config = {

    # label
    "is_click": tf.io.FixedLenFeature([], tf.int64)
}

data_config = {
    "rank-dumn": rank_dumn_config
}
