# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/10/23
# Company : Maxent
# Email: chao.xu@maxent-inc.com

"""
tis file used to draw precision and recall curve
"""
from __future__ import print_function, division
from matplotlib import pylab as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score
from Pic.maxent_style import maxent_style,remove_palette
import numpy as np
from adjustText import adjust_text
import pandas as pd


@maxent_style
def pic_pr_bin_fill(y_true,probs, title, file_path,dpi=600,palette=None):
    average_precision = average_precision_score(y_true=y_true,y_score=probs)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=probs[:,1])

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.step(recall, precision, color=next(palette), where='post', alpha=0.2)
    ax.fill_between(recall, precision, color=next(palette), step='post', alpha=0.2)
    title += " with APR is {0:0.2f}".format(average_precision)
    ax.set_title(title)
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)


@maxent_style
def pic_pr_bin_threshold(y_true,probs, title, file_path,dpi=600,palette=None):
    average_precision = average_precision_score(y_true=y_true,y_score=probs)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    precision, recall, threshold = precision_recall_curve(y_true=y_true, probas_pred=probs[:,1])
    threshold = np.append(threshold, 1)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(threshold, precision, label="Precision according threshold",color=next(palette), marker="o",ms=2,linestyle="--")
    ax.plot(threshold, recall, label="Recall according threshold",color=next(palette), marker="o",ms=2,linestyle="--")
    title += " with APR is {0:0.2f}".format(average_precision)
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.set_ylabel("precision/recall")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)


@maxent_style
def pic_pr_cmp_threshold(title, file_path,dpi=600,palette=None, y_data=[]):
    """
    :param threshold: the threshold change to see the precision and recall
    :param title:  fig title
    :param file_path:
    :param dpi:
    :param palette:
    :param y_data: data structure is [(label,y_true,y_pred),...]
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    for label,y_true,y_prob in y_data:
        precision, recall, threshold = precision_recall_curve(y_true, y_prob)
        threshold = np.append(threshold, 1)
        color = next(palette)
        ax.plot(threshold, recall, label=label + " recall", color=color, marker="o",ms=2,linestyle="--")
        ax.plot(threshold, precision, label=label + " precision", color=color, marker="o",ms=2,linestyle="-")
    ax.set_title(title)
    # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_xlabel("threshold")
    ax.set_ylabel("precision/recall")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)


@maxent_style
def pic_pr_cmp(title, file_path,dpi=600,palette=None, y_data=[]):
    """
    this function used to compare the precsion and recall curve
    :param title:
    :param file_path:
    :param dpi:
    :param palette:
    :param y_data:
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    texts = []
    add_objs = []
    for label,y_true,y_prob in y_data:
        precision, recall, threshold = precision_recall_curve(y_true, y_prob)
        line = ax.plot(recall, precision, label=label + " PR Curve", color=next(palette), marker="o",ms=2,linestyle="-")
        threshold = np.append(threshold, 1)
        abs_pr = np.abs(precision - recall)
        luck_index = np.argmin(abs_pr)
        luck_recall = recall[luck_index]
        luck_precision = precision[luck_index]
        luck_threshold = threshold[luck_index]
        # bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        # arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        # kw = dict(xycoords='data', textcoords="axes fraction",
        #           arrowprops=arrowprops, bbox=bbox_props, ha="left", va="center")
        #
        # # arrowprops = dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        # # kw = dict(xycoords='data', textcoords="offset points",
        # #           arrowprops=arrowprops, ha="left", va="bottom")
        # ax.annotate("threshold={0:.3f}".format(luck_threshold),
        #             xy=(luck_recall, luck_precision),
        #             xytext=(0.5, 0.5),
        #             **kw)
        texts.append(ax.text(luck_recall, luck_precision, "{0}:\nBEP={1:.3f}\nthreshold:{2:.3f}".format(label,luck_recall,luck_threshold)))
        add_objs.extend(line)
    luck_line = ax.plot([0, 1], [0, 1], label="BEP", color=next(palette), marker="x", linestyle="--")
    add_objs.extend(luck_line)
    adjust_text(texts=texts,
                # add_objects=add_objs,
                autoalign='xy', expand_objects=(0.1, 1),
                text_from_points=False,
                text_from_text=False,
                only_move={'text': 'y', 'objects': 'y'}, force_text=1.2, force_objects=0.5,
                arrowprops=dict(arrowstyle="simple, head_width=0.25, tail_width=0.1", color=next(palette), lw=0.5, alpha=0.8))
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)


@maxent_style
def pic_pr_step_time(title, file_path, dpi=600, delta=5,palette=None, y_data=[]):
    """
    :param title:
    :param file_path:
    :param threshold:
    :param dpi:
    :param delta: time delta for precision and recall sample
    :param palette:
    :param y_data:data structure is [(label,y_true,y_pred, time_diff, threshold),...], y_true, y_test, time_diff must be seriies
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    def group_precision(x):
        score = precision_score(x['true'],x['pred'])
        return score

    def group_recall(x):
        score = recall_score(x['true'],x['pred'])
        return score

    for label,y_true,y_pred,time_delta,threshold in y_data:
        # data = pd.DataFrame([y_true, y_pred, time_delta],columns=['true', 'pred', 'delta'])
        data = pd.DataFrame.from_items(zip(('true', 'pred', 'delta'), (y_true, y_pred, time_delta)))
        data=data.sort(['delta'],axis=0)
        time_min = data['delta'].min()
        time_max = data['delta'].max()
        bins = np.arange(time_min,time_max,delta)
        x_bins = pd.cut(data['delta'],bins=bins)
        df_group = data.groupby(x_bins)
        precision_df = df_group.apply(lambda x:group_precision(x)).reset_index(name='precision')
        recall_df = df_group.apply(lambda x:group_recall(x)).reset_index(name='recall')
        df = pd.merge(precision_df,recall_df,on='delta')
        x = df.delta.str.extract('.*,\s*(\d+.\d+)]', expand=False)
        color = next(palette)
        ax.plot(x,df['precision'], c=color, label="{0} precision:\nthreshold:{1:.3f}".format(label,threshold),marker="o",ms=2,linestyle="-")
        ax.plot(x,df['recall'], c=color, label="{0} recall:\nthreshold:{1:.3f}".format(label,threshold),marker="o",ms=2,linestyle="--")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel("time step: {0}".format(delta))
    ax.set_ylabel("Precision/Recall")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)


@maxent_style
def pic_pr_travel_time(title, file_path,dpi=600, delta=5,palette=None, y_data=[]):
    """
    this function used to draw precision and recall increase time
    :param title:
    :param file_path:
    :param threshold:
    :param dpi:
    :param delta:
    :param palette:
    :param y_data:data structure is [(label,y_true,y_pred, time_diff),...]
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    for label,y_true,y_pred,time_delta,threshold in y_data:
        data = pd.DataFrame.from_items(zip(('true', 'pred', 'delta'), (y_true, y_pred, time_delta)))
        data=data.sort(['delta'],axis=0)
        time_min = data['delta'].min()
        time_max = data['delta'].max()
        bins = np.arange(time_min,time_max,delta)
        precisions = []
        recalls = []
        for x in bins:
            y_true = data.loc[data.delta <= x]['true']
            y_test = data.loc[data.delta <= x]['pred']
            precision = precision_score(y_true,y_test)
            recall = recall_score(y_true,y_test)
            precisions.append(precision)
            recalls.append(recall)
        color = next(palette)
        ax.plot(bins,precisions, c=color, label="{0} precision:\nthreshold:{1:.3f}".format(label,threshold),marker="o",ms=2,linestyle="-")
        ax.plot(bins,recalls, c=color, label="{0} recall:\nthreshold:{1:.3f}".format(label,threshold),marker="o",ms=2,linestyle="--")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel("time increase {0}".format(delta))
    ax.set_ylabel("Precision/Recall")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
    fig.subplots_adjust(left=0.1,right=0.7)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)

