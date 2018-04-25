# -*- coding: utf-8 -*-
# Project: maxent-ml
# Author: chaoxu create this file
# Time: 2017/9/7
# Company : Maxent
# Email: chao.xu@maxent-inc.com

from __future__ import print_function, division
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pylab as plt
import itertools
from Pic.maxent_style import maxent_style, remove_palette
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(cm,
                          ax,
                          fig,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.set_title(title)
    # ax.colorbar()
    ax.set_title(title)
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar

    ax.set_xticks(classes)
    ax.set_yticks(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('test label')
    ax.set_xlabel('predict label')


@maxent_style
# @remove_palette
def plot_confusion_matrix_sns(y_pred,
                              y_test,
                              title,
                              path,
                              threshold,
                              palette=None,
                              dpi=600):
    """
    this function use seaborn to print better confusion matrix
    :param y_pred:
    :param y_test:
    :param title:
    :param path:
    :param dpi:
    :return:
    """
    cm_data = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    class_names = [0, 1]
    # grid_kws = {"height_ratios": (.9, .05), "hspace": .05}
    # fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,figsize=(12, 6))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    df_cm = pd.DataFrame(
        cm_data, index=class_names, columns=class_names,
    )
    # df_cm_normal = df_cm / df_cm.sum()
    cmap = sns.light_palette(next(palette), as_cmap=True)
    ax = sns.heatmap(df_cm,
                     ax=ax,
                     cmap=cmap,
                     annot=True,
                     cbar=False,
                     linewidths=.3,
                     fmt='d'
                     )
    text = classification_report(y_true=y_test, y_pred=y_pred)
    text_print = "Threshold is {0}: \n{1}".format(threshold, text)
    ax.text(2.05, 1, text_print, ha='left', va='center')
    fig.subplots_adjust(left=0.1, right=0.7)
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right')
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    path += "/{0}.png".format(title.replace(" ", "_"))
    fig.savefig(filename=path, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
@remove_palette
def confusion_one(y_pred, y_test, title, path, dpi=600, block=False):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plot_confusion_matrix(cnf_matrix, ax=ax, fig=fig,
                          classes=class_names,
                          title=title)
    path += "/{0}.png".format(title.replace(" ", "_"))
    fig.savefig(filename=path, dpi=dpi, format='png')
    plt.show(block=block)


@maxent_style
@remove_palette
def confusion_threshold(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train.values.ravel())
    y_or_pred_proba = model.predict_proba(X_test.values)
    thresholds = np.arange(0.1, 1, 0.1)
    j = 1
    for i in thresholds:
        print("threshold is ", i)
        y_or_test_predictions_high_recall = y_or_pred_proba[:, 1] > i

        plt.subplot(3, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_or_test_predictions_high_recall)
        np.set_printoptions(precision=2)

        print("precision metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1]))
        print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

        # Plot non-normalized confusion matrix
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix
                              , classes=class_names
                              , title='Threshold >= %s' % i)
    plt.show()
