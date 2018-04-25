# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/10/23
# Company : Maxent
# Email: chao.xu@maxent-inc.com

"""
this file is used to draw roc picture
"""

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from Pic.maxent_style import maxent_style
import matplotlib.pyplot as plt


@maxent_style
def pic_roc(y_test, probs, title, file_path, dpi=600, palette=None):
    fpr, tpr, thresholds = roc_curve(y_test, y_score=probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(fpr, tpr, label="AUC = {0:0.3f}".format(roc_auc), color=next(palette), marker="o", linestyle="--")
    ax.plot([0, 1], [0, 1], label="Luck", color="r", marker="o", ms=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1, right=0.7)
    file_path += "/{0}.png".format(title.replace(" ", "_"))
    fig.savefig(filename=file_path, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def pic_roc_cmp(title, file_path, dpi=600, palette=None, y_data=[]):
    """
    this function use to plot roc comparation
    :param title:
    :param file_path:
    :param dpi:
    :param palette:
    :param y_data:
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    for label, y_true, y_prob in y_data:
        fpr, tpr, threshold = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=label + " AUC = {0:0.3f}".format(roc_auc), color=next(palette), marker="o", ms=2,
                linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ", "_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1, right=0.7)
    fig.savefig(filename=file_path, dpi=dpi, format='png')
    plt.show(block=False)
