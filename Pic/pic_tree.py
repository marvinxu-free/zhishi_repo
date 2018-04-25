# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/11/8
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
this file used to draw decision tree
"""
import pydotplus
from sklearn import tree
from Pic.maxent_style import maxent_style,remove_palette
from Pic.maxent_font import tick_xfont
from Pic.pic_score import plot_learning_curves
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns



def pic_tree(clf, fcols, title, file_path,dpi=600):
    """

    :param clf:must be decision tree model
    :param fcols: feature columns
    :param title:
    :param file_path:
    :param dpi:
    :return:
    """
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=fcols, class_names=['label 0','label 1'],\
                               filled=True, rounded=True, special_characters=True, leaves_parallel=False)
    graph = pydotplus.graph_from_dot_data(dot_data)
    file_name = file_path + "/{0}.png".format(str(title).replace(" ","_"))
    graph.write_png(file_name)


@maxent_style
@remove_palette
def pic_performence(clf, X_train, y_train, X_test, y_test, tareget='label',
                    file_path=None, dpi=600):
    """
    plot performence
    :param clf: ml model
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param target: classification column
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    y_train_f = y_train[tareget].astype(float)
    y_test_f = y_test[tareget].astype(float)
    ax, errors = plot_learning_curves(X_train=X_train, y_train=y_train_f, X_test=X_test, y_test=y_test_f,
                         clf=clf, ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    title = ax.get_title()
    file_path += "/{0}.png".format(title.replace(" ","_"))
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)


@maxent_style
@remove_palette
def pic_feature_importance(clf, cols, title, file_path=None, dpi=600):
    """
    this function used to plot feature importance
    :param clf:
    :param cols:
    :param title:
    :param file_path:
    :param dpi:
    :return:
    """
    df_feature = pd.DataFrame(cols, columns=['features'])
    df_importance = pd.DataFrame(clf.feature_importances_, columns=['importances'])
    df_feature_importance = pd.concat([df_feature, df_importance], axis=1)
    df_feature_importance = df_feature_importance.sort(['importances'],ascending=False)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    # ax = sns.barplot(x='features',y='importances',data=df_feature_importance,ax=ax)
    for x, y in enumerate(df_feature_importance['importances'].values):
        ax.bar(x, y)
    x_labels = df_feature_importance['features'].values
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    tick_xfont(ax,rotation=90)
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    file_path += "/{0}.png".format(title.replace(" ","_"))
    fig.subplots_adjust(bottom=0.28, top=0.88)
    fig.savefig(filename=file_path,dpi=dpi,format='png')
    plt.show(block=False)
