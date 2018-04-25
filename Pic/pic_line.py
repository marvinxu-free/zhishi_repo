# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/12/11
# Company : Maxent
# Email: chao.xu@maxent-inc.com
from __future__ import print_function, division
import matplotlib.pyplot as plt
from maxent_style import maxent_style
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')


@maxent_style
def pic_line_threshold_ratio(thresholds, fname, title, dpi=600, palette=None, reverse=True):
    """
    this function use to split threshold to num scopes and find if scope < thresholds' ratio
    all input data should be ndarray
    :param x:
    :param y:
    :param fname:
    :param dpi:
    :param palette:
    :return:
    """
    thresholds_max = thresholds.max()
    thresholds_min = thresholds.min()
    thresholds_sum = thresholds.sum()
    if thresholds_max <= 100:
        steps = np.arange(thresholds_min,thresholds_max,1)
    else:
        num = int(thresholds_max / 100)
        steps = np.arange(thresholds_min, thresholds_max, num)
    ratios = []
    for step in steps:
        if reverse:
            ratio = thresholds[thresholds >= step].sum() / thresholds_sum
        else:
            ratio = thresholds[thresholds <= step].sum() / thresholds_sum
        ratios.append(ratio)
    ratios = np.array(ratios)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, ratios, c=next(palette),marker="^",
            ms=2, linestyle="-")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("调整数值")
    ax.set_ylabel("影响用户占比")
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def pic_line_threshold_user_ratio(thresholds, users, fname, title, dpi=600, palette=None, reverse=True):
    """

    :param thresholds:
    :param users:
    :param fname:
    :param title:
    :param dpi:
    :param palette:
    :param reverse:
    :return:
    """
    thresholds_max = thresholds.max()
    thresholds_min = thresholds.min()
    user_sum = users.sum()
    if thresholds_max <= 100:
        steps = np.arange(thresholds_min,thresholds_max,1)
    else:
        num = int(thresholds_max / 100)
        steps = np.arange(thresholds_min, thresholds_max, num)
    ratios = []
    for step in steps:
        if reverse:
            ratio = users[thresholds >= step].sum() / user_sum
        else:
            ratio = users[thresholds <= step].sum() / user_sum
        ratios.append(ratio)
    ratios = np.array(ratios)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, ratios, c=next(palette),marker="^",
            ms=2, linestyle="-")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("调整数值")
    ax.set_ylabel("影响用户占比")
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)
