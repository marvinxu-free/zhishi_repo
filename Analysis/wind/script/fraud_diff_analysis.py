# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/24
# Company : Maxent
# Email: chao.xu@maxent-inc.com
from matplotlib import pyplot as plt
import pandas as pd
from Params.path_params import *
# from Pic.hist import hist_with_poisson
from Pic.pic_wind import fig_one_col


def diff_analysis():
    """
    use to analysis fraud diff distribution analysis
    :param xls_file:
    :return:
    """
    lof_xls_file = f'{Data_path}/wind/你只是不懂LOF.xlsx'
    # close_xls_file = f'{Data_path}/wind/上市封闭式基金.xlsx'

    print(lof_xls_file)

    df_lof = pd.read_excel(lof_xls_file, sheet_name=0)
    print('LOF基金统计\n', df_lof[['贴水', '贴水率']].describe())
    lof_img_name = f'{Data_path}/wind/LOF(场内)贴水率轴须图.png'
    lof_title = '2018年4月24日LOF(场内)基金贴水率轴须图'
    fig_one_col(df=df_lof, col='贴水率', title=lof_title, img_name=lof_img_name)
    # hist_with_poisson(col='贴水率', df=df_lof, title=lof_title, fname=lof_img_name)

    lof1_title = '2018年4月24日LOF(场内)基金贴水轴须图'
    lof1_img_name = f'{Data_path}/wind/LOF(场内)贴水轴须图.png'
    fig_one_col(df=df_lof, col='贴水', title=lof1_title, pecent=False, img_name=lof1_img_name)

    df_close = pd.read_excel(lof_xls_file, sheet_name=1)
    print('封闭式基金统计\n', df_close[['贴水', '贴水率']].describe())
    close_img_name = f'{Data_path}/wind/上市封闭式基金贴水率轴须图.png'
    close_title = '2018年4月24日上市封闭式基金贴水率轴须图'
    fig_one_col(df=df_close, col='贴水率', title=close_title, img_name=close_img_name)

    close1_title = '2018年4月24日上市封闭式基金贴水轴须图'
    close1_img_name = f'{Data_path}/wind/上市封闭式基金贴水轴须图.png'
    fig_one_col(df=df_close, col='贴水', title=close1_title, pecent=False, img_name=close1_img_name)


if __name__ == '__main__':
    diff_analysis()
