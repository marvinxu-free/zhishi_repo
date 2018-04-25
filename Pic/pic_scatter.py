# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/12/13
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
this file used to pic scatter figure
"""
from __future__ import division, print_function
from maxent_style import maxent_style
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.dates as dates
from datetime import datetime
import matplotlib.ticker as ticker
import pandas as pd


@maxent_style
def pic_scatter_bubble(x, y,z, text, title, fname, xlabel, ylabel, zlabel, dpi=600, palette=None):
    """
    this function used to pic a bubble figure
    :param x:date
    :param y:value
    :param z: ratio
    :param titile:
    :param path:
    :param dpi:
    :param palette:
    :return:
    """
    # date_formater = DateFormatter('%Y-%m-%d')
    # days = DayLocator()
    area_scale, width_scale = 500, 5
    fig = plt.figure(figsize=(12, 6))
    ax = Axes3D(fig)
    x_int = x.astype(datetime)
    # x_days = map(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"), x)
    # def format_date(x, pos=None):
    #     return dates.num2date(x).strftime('%Y-%m-%d-%h:%s:%ms:%ns')

    ax.scatter(xs=x_int, ys=y, zs=z, s=np.square(z) * area_scale, c=next(palette))
    map(lambda x : ax.text(x[0], x[1], x[2], x[3]), zip(x_int, y, z, text))
    # for x_, y_, z_, text_ in zip(x_int, y, z, text):
    #     ax.text(x_, y_, z_, text_)
    # ax.w_xaxis.set_major_locator(ticker.FixedLocator(x_int))  # I want all the dates on my xaxis
    # ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    # for tl in ax.w_xaxis.get_ticklabels():
    #     tl.set_ha('right')
    #     tl.set_rotation(30)
    ax.set_title(title)
    ax.xaxis.set_ticks([])
    # ax.xaxis.set_ticks(x_int)
    # ax.xaxis.set_ticklabels(x_days,rotation=90)
    # ax.set_xlim(x_int.min() * 0.9, x_int.max() * 1.1)
    if xlabel:
        ax.set_xlabel(xlabel=xlabel)
    if ylabel:
        ax.set_ylabel(ylabel=ylabel)
    if zlabel:
        ax.set_zlabel(zlabel=zlabel)
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def pic_scatter_bubble_2d(x, y,text, title=None, fname=None, xlabel=None, ylabel=None, dpi=600, palette=None):
    area_scale, width_scale = 500, 5
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=x, y=y, s=np.square(y) * area_scale, c=next(palette))
    map(lambda x : ax.text(x[0], x[1], x[2]), zip(x, y, text))
    if title:
        ax.set_title(title)
    # ax.xaxis.set_ticklabels(x_days,rotation=90)
    # ax.set_xlim(x_int.min() * 0.9, x_int.max() * 1.1)
    if xlabel:
        ax.set_xlabel(xlabel=xlabel)
    if ylabel:
        ax.set_ylabel(ylabel=ylabel)
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)


