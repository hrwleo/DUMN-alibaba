import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff, axis=-1)


def categorical_ce(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def sparse_ce(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)


def bce_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def ctr_loss(y_true, y_pred):
    ctr_true = tf.cast(tf.slice(y_true, [0, 0], [-1, 1]), 'float32')
    ctr_pred = tf.cast(tf.slice(y_pred, [0, 0], [-1, 1]), 'float32')
    return bce_loss(ctr_true, ctr_pred)


def ctcvr_loss(y_true, y_pred):
    ctcvr_true = tf.cast(tf.slice(y_true, [0, 1], [-1, 1]), 'float32')
    ctcvr_pred = tf.cast(tf.slice(y_pred, [0, 1], [-1, 1]), 'float32')
    return bce_loss(ctcvr_true, ctcvr_pred)


def mmoe_total_loss(y_true, y_pred):
    ctrloss = ctr_loss(y_true, y_pred)
    ctcvrloss = ctcvr_loss(y_true, y_pred)

    return 0.5 * (ctrloss + ctcvrloss)


def ctr_auc(y_true, y_pred):
    ctr_true = tf.cast(tf.slice(y_true, [0, 0], [-1, 1]), 'float32')
    ctr_pred = tf.cast(tf.slice(y_pred, [0, 0], [-1, 1]), 'float32')
    return auc(ctr_true, ctr_pred)

def ctcvr_auc(y_true, y_pred):
    ctcvr_true = tf.cast(tf.slice(y_true, [0, 1], [-1, 1]), 'float32')
    ctcvr_pred = tf.cast(tf.slice(y_pred, [0, 1], [-1, 1]), 'float32')
    return auc(ctcvr_true, ctcvr_pred)


# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P
