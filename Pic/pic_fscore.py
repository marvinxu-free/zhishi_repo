# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/10/23
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
this file used to draw fscore curve
"""
from __future__ import print_function, division
from Pic.maxent_style import maxent_style,remove_palette
from sklearn.metrics import precision_recall_curve
import matplotlib.pylab as plt
import numpy as np
from adjustText import adjust_text


def f1_score_prob(precision, recall):
    """
    this function use to caculate f1 score
    :param precision:
    :param recall:
    :return:
    """
    f1_scores = []
    for x, y in zip(precision,recall):
        f1 = 2.0 * (x * y) / (x + y)
        f1_scores.append(f1)
    return f1_scores

def fbeta_score_prob(precision, recall,beta):
    """
    this function use to caculate f1 score
    :param precision:
    :param recall:
    :return:
    """
    fbeta_scores = []
    beta_2 = 1.0 * beta * beta
    for x, y in zip(precision,recall):
        if x == 0 and y == 0:
            fbeta = 0
        else:
            fbeta = 1.0 * (1 + beta_2) * x * y / ((beta_2 * x) + y)
        fbeta_scores.append(fbeta)
    return fbeta_scores

@maxent_style
def f1_score_bin_prob(y_true,probs, title, file_path,dpi=600,palette=None):
    """
    this function use difference prob get get precision and recall first and then caculate f1 score
    :param y_true:
    :param probs:
    :param title:
    :param file_path:
    :param dpi:
    :param palette:
    :return:
    """
    precision, recall, threshold = precision_recall_curve(y_true=y_true, probas_pred=probs)
    threshold = np.append(threshold, 1)
    f1_score = f1_score_prob(precision,recall)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(threshold, f1_score, label="f1_score according threshold",color=next(palette), marker="o",ms=2,linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.set_ylabel("f1 score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)


@maxent_style
def pic_f1_cmp(title, file_path,dpi=600,palette=None, y_data=[]):
    """
    :param threshold: the threshold change to see the precision and recall
    :param title:  fig title
    :param file_path:
    :param dpi:
    :param palette:
    :param y_data: data structure is [(label,y_true,y_prob)...]
    :return: nothing
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    texts = []
    add_objs = []
    best_thresholds = {}
    for label,y_true,y_prob in y_data:
        precision, recall, threshold = precision_recall_curve(y_true, y_prob)
        threshold = np.append(threshold, 1)
        f1_scores = f1_score_prob(precision,recall)
        f1_scores = np.array(f1_scores)
        line = ax.plot(threshold, f1_scores, label=label, color=next(palette), marker="o",ms=2,linestyle="--")
        f1_max = f1_scores.max()
        max_index = np.argmax(f1_scores)
        best_threshold = threshold[max_index]
        best_thresholds[label] = best_threshold
        texts.append(ax.text(best_threshold, f1_max,
                              "{0}:\nthreshold={1:.2f}\nf1={2:.2f}\nprecision={3:.2f}\nrecall={4:.2f}"\
                             .format(label,best_threshold, f1_max, precision[max_index], recall[max_index])))
        add_objs.extend(line)
    adjust_text(texts=texts,
                # add_objects=add_objs,
                autoalign='xy', expand_objects=(0.1, 1),
                text_from_points=True,
                text_from_text=False,
                only_move={'text': 'y', 'objects': 'y'}, force_text=1.2, force_objects=0.5,
                arrowprops=dict(arrowstyle="simple, head_width=0.25, tail_width=0.1", color=next(palette), lw=0.5, alpha=0.8))
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.set_ylabel("f1 score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)
    return best_thresholds


@maxent_style
def pic_fbeta_cmp(title, file_path,dpi=600,palette=None, beta=0.5, y_data=[]):
    """
    :param threshold: the threshold change to see the precision and recall
    :param title:  fig title
    :param file_path:
    :param dpi:
    :param palette:
    :param y_data: data structure is [(label,y_true,y_prob)...]
    :return: nothing
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    texts = []
    add_objs = []
    best_thresholds = {}
    for label,y_true,y_prob in y_data:
        precision, recall, threshold = precision_recall_curve(y_true, y_prob)
        threshold = np.append(threshold, 1)
        fbeta_scores = fbeta_score_prob(precision,recall,beta=beta)
        fbeta_scores = np.array(fbeta_scores)
        line = ax.plot(threshold, fbeta_scores, label=label, color=next(palette), marker="o",ms=2,linestyle="--")
        f1_max = fbeta_scores.max()
        max_index = np.argmax(fbeta_scores)
        best_threshold = threshold[max_index]
        best_thresholds[label] = best_threshold
        texts.append(ax.text(best_threshold, f1_max,
                             "{0}:\nthreshold={1:.2f}\nbeta={2:.1f}\nfbeta={3:.2f}\nprecision={4:.2f}\nrecall={5:.2f}" \
                             .format(label,best_threshold, beta, f1_max, precision[max_index], recall[max_index])))
        add_objs.extend(line)
    adjust_text(texts=texts,
                add_objects=add_objs,
                autoalign='xy', expand_objects=(0.1, 1),
                text_from_points=True,
                text_from_text=False,
                only_move={'text': 'y', 'objects': 'y'}, force_text=1.2, force_objects=0.5,
                arrowprops=dict(arrowstyle="simple, head_width=0.25, tail_width=0.1", color=next(palette), lw=0.5, alpha=0.8))
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.set_ylabel("fbeta score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)
    return best_thresholds
