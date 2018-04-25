# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/8/17
# Company : Maxent
# Email: chao.xu@maxent-inc.com

def tick_font(ax=None,font_size=None,rotation=90,rotation_y=90):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)
        tick.label.set_rotation(rotation)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)
        tick.label.set_rotation(rotation_y)
    return None


def tick_xfont(ax=None,font_size=None,rotation=90):
    for tick in ax.xaxis.get_major_ticks():
        if font_size is not None:
            tick.label.set_fontsize(font_size)
        if rotation is not None:
            tick.label.set_rotation(rotation)
    return None
