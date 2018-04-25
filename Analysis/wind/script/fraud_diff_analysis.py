# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/24
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import pandas as pd
from Params.path_params import *
from matplotlib import pyplot as plt
import seaborn as sns
from Pic.maxent_style import maxent_style
from scipy.stats import norm
import numpy as np


@maxent_style
def fig_one_col(df, col, title, img_name, dpi=600):
    """
    本函数用于绘制单列的分布图
    :param df:
    :param col:
    :param palette:
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax=sns.distplot(ax=ax, a=df[col].values, rug=True, hist=True, color='salmon', norm_hist=True)
    # ax.set_title(title)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    # label_0_index = np.where(labels == 0 or labels == '0')
    ax.set_ylabel('基金数量')
    # ax.set_xlim(df[col].min() - 0.1, df[col].max() + 0.1)
    # fig.canvas.set_window_title(title)
    fig.savefig(fname=img_name, dpi=dpi, format='png')
    plt.show(block=False)


def diff_analysis():
    """
    use to analysis fraud diff distribution analysis
    :param xls_file:
    :return:
    """
    lof_xls_file = f'{Data_path}/wind/LOF(场内).xlsx'
    close_xls_file = f'{Data_path}/wind/上市封闭式基金.xlsx'

    lof_img_name = f'{Data_path}/wind/LOF(场内)差价轴须图.png'
    close_img_name = f'{Data_path}/wind/上市封闭式基金.png'
    print(lof_xls_file, close_xls_file)

    df = pd.read_excel(lof_xls_file, sheet_name=2)
    df_new = df[['差价']]
    print(df_new.describe())
    lof_title = '2018年4月24日LOF(场内)基金差价轴须图'
    fig_one_col(df=df_new, col='差价', title=lof_title, img_name=lof_img_name)

    df = pd.read_excel(close_xls_file, sheet_name=2)
    df_new = df[['差价']]
    print(df_new.describe())
    close_title = '2018年4月24日上市封闭式基金差价轴须图'
    fig_one_col(df=df_new, col='差价', title=close_title, img_name=close_img_name)


if __name__ == '__main__':
    diff_analysis()
