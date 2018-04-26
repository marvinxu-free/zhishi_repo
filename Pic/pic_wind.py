from Pic.maxent_style import maxent_style
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np


@maxent_style
def fig_one_col(df, col, title, img_name, pecent=True, dpi=600):
    """
    本函数用于绘制单列的分布图
    :param df:
    :param col:
    :param palette:
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.distplot(a=df[col],
                      kde=True,
                      # fit=norm,
                      rug=True,
                      color='salmon',
                      ax=ax,
                      )
    # density, bins, patches = plt.hist(df[col].values, bins=50, density=False)
    ax.set_title(title)
    # ax.set_ylabel('基金数量')
    ax.set_xlabel(f'{col}')
    if pecent:
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:3.2f}%'.format(x) for x in vals])
    fig.savefig(fname=img_name, dpi=dpi, format='png')
    plt.show(block=False)
    plt.close()
