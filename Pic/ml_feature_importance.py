# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2018/2/1
# Company : Maxent
# Email: chao.xu@maxent-inc.com

from xgboost import plot_importance
from matplotlib import pyplot as plt
from Pic.maxent_style import maxent_style


@maxent_style
def plot_xgb_importance(model, fname, title=None, dpi=600, palette=None):
    """

    :param model:
    :param fname:
    :param dpi:
    :param palette:
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_importance(booster=model, ax=ax)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0.2, right=0.95)
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)
