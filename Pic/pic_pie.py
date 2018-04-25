# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/12/12
# Company : Maxent
# Email: chao.xu@maxent-inc.com
from maxent_style import maxent_style
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm



@maxent_style
def pic_pie(values, cats, title, fname=None, dpi=280, palette=None):
    """
    this function used to pic a pie chart
    :param values:
    :param cats:
    :param title:
    :param fname:
    :param path: should be as abs file with .png
    :param palette:
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(1, 1, 1)
    """
    (x0, y0, width, height)
    x0, y0定义新的 ax的左下角坐标
    height 与width是在x0, y0基础上，然后scale对应比例长度的的坐标轴
    1. 假设x0缩放比例为a， width缩放比例为b，不考虑Y轴缩放的情况下：
    要满足pie图中心在中点，那么有a*x + 0.5 *b *x = 0.5 *x
    也就是2a + b = 1
    """
    ax = fig.add_axes((0.25, 0, 0.5, 1))
    patches, texts, autotexts = ax.pie(values, colors=palette, shadow=True,
                                       labels=cats, autopct='%1.1f%%', labeldistance=1.05)
    for pie_wedge in patches:
        pie_wedge.set_edgecolor('white')
    map(lambda x: x.set_fontsize(10), texts)
    map(lambda x: x.set_fontsize(10), autotexts)
    ax.set_title(title, fontdict={"fontsize": 12})
    fig.canvas.set_window_title(title)
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)

