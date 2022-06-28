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
    "user_city_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_shangquan_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_comm_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_price_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_area_seq": tf.io.FixedLenFeature([5], tf.int64),

    # prop
    "city_id": tf.io.FixedLenFeature([1], tf.int64),
    "comm_id": tf.io.FixedLenFeature([1], tf.int64),
    "shangquan_id": tf.io.FixedLenFeature([1], tf.int64),
    "price_id": tf.io.FixedLenFeature([1], tf.int64),
    "area_id": tf.io.FixedLenFeature([1], tf.int64),
    "floor_loc": tf.io.FixedLenFeature([1], tf.int64),
    "total_floor": tf.io.FixedLenFeature([1], tf.int64),
    "room_num": tf.io.FixedLenFeature([1], tf.int64),
    "hall": tf.io.FixedLenFeature([1], tf.int64),
    "bathroom": tf.io.FixedLenFeature([1], tf.int64),
    "prop_age": tf.io.FixedLenFeature([1], tf.float32),

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
