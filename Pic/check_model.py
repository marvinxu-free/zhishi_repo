# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/8/18
# Company : Maxent
# Email: chao.xu@maxent-inc.com
from __future__ import division
from scipy.special import gammainc
from scipy.stats import norm
from math import ceil, sqrt
import numpy as np
import pandas as pd
import math


def poison_p(pi, _lambda, k, m, v):
    """
    this function is used to compute p
    :param pi: model pi from mysql for each event
    :param k: count value 
    :return: p value 
    """
    if _lambda <= float("-inf") and pi <= float("-inf") and m <= float("-inf") and v <= float("-inf"):
        p = 1.0
    elif _lambda > 20 and m > 0:
        p1 = calP(pi, _lambda, k)
        p2 = calPNor(m, v, k)
        p = max(p1, p2)
    else:
        p = calP(pi, _lambda, k)
    return p


def calP(pi, _lambda, k):
    """
    this function is used to calculate p value which 
    :param pi: 
    :param _lambda: 
    :param k: 
    :return: 
    """
    if k < 2:
        p = 1.0
    elif pi == 1.0:
        p = 0.0001
    else:
        p = (1 - pi) * (gammainc(int(k - 1), _lambda))
    return p


def calPNor(m, v, k):
    """
    this function is used to calculate p which is normal
    distribution
    :param m: 
    :param v: 
    :param k: 
    :return: 
    """
    if (v <= 0):
        p = 0.0001
    else:
        # p = 1 - norm(m,v).cdf(k)
        p = cumulativeProbability(k, m, v)
    return p


def cumulativeProbability(x, m, v):
    dev = x - m
    sigma = math.sqrt(v)
    if math.fabs(dev) > 40 * sigma:
        if dev < 0:
            p = 0.0
        else:
            p = 1.0
    else:
        p = 0.5 * math.erfc(-dev / (sigma * math.sqrt(2)))
    return 1 - p


def maxent_fs(p_value):
    """
    this function is used to compute FS
    :param p_value: event variable happen possibility
    :return: FS value
    """
    if p_value == 0:
        fs = 1000
    else:
        fs = ceil(min(sqrt(1 / p_value) / 10, 1000))
    return fs


def check_model(df, df1, feature="ipGeo", interval="1d", tid="Q4R5AM9UNIH9RCSYHALWUFOTDKTMDT8H", short=False):
    feature_value = "{0}.{1}.{2}".format(feature, interval, "value")
    feature_anomaly = "{0}.{1}.{2}".format(feature, interval, "anomaly")
    uncheck = df.loc[(df[feature_value] == df[feature_value].max())]
    k = uncheck[feature_value].values[0]
    fs_check = uncheck[feature_anomaly].values[0]
    event_type = uncheck['event_type'].values[0]
    if short is False:
        col_value = uncheck[feature_value].values[0]
        model = df1.loc[((df1['col_type'] == feature) & (df1['col_value'] == col_value) \
                         & (df1['time_interval'] == interval) & (df1['event_type'] == event_type) \
                         & (df1['tid'] == tid))]
        if (model["pi"].values.size == 0):
            p = 1
            print("check {0}:{1} passed".format([k, event_type, col_value, p], fs_check))
        else:
            pi = model["pi"].values[0]
            _lambda = model["lambda"].values[0]
            m = model["m"].values[0]
            v = model["variance"].values[0]
            p = poison_p(pi=pi, _lambda=_lambda, k=k, m=m, v=v)
            print("check {0}:{1} ".format([k, event_type, col_value, pi, _lambda, m, v], fs_check))
    else:
        model = df1.loc[((df1['col_type'] == feature) \
                         & (df1['time_interval'] == interval) & (df1['event_type'] == event_type) \
                         & (df1['tid'] == tid))]
        if (model["pi"].values.size == 0):
            p = 1
            print("check {0}:{1} passed".format([k, event_type, p], fs_check))
        else:
            pi = model["pi"].values[0]
            _lambda = model["lambda"].values[0]
            p = calP(pi=pi, _lambda=_lambda, k=k)
            print("check {0}:{1} ".format([k, event_type, pi, _lambda], fs_check))
    fs = maxent_fs(p)
    print("check fs:fs_check score {0}:{1}:{2}".format(fs, fs_check, p))
    if fs == fs_check:
        print("check passed")
    else:
        print("check failed")
    assert fs == fs_check


def scored(score_init, num):
    if score_init <= 0:
        score = 0
    else:
        score_avg = score_init / num
        score1 = (1 - 7 / score_init) / 0.86 * 1.35 + 0.85
        score2 = (1 - 0.22 / score_avg) / (2.78 / 3) * 1.35 + 0.85
        max_score = max(score1, score2)
        score_e = 1 + np.exp(-max_score)
        score = int(1 / score_e * 100)
    return score
