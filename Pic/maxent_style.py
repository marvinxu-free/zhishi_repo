import matplotlib
import seaborn as sns
import itertools

matplotlib.pyplot.switch_backend('agg')


def maxent_style(func):
    def maxent_style_inner(*args, **kwargs):
        print(func.__name__)
        with sns.axes_style('whitegrid', {"axes.grid": True, "axes.edgecolor": '.8'}):
            # with sns.axes_style('ticks', {'grid.color': '.8', 'grid.linestyle': u'-'}):
            # with sns.axes_style('ticks',{'grid.color': 'white'}):
            sns.set_palette('muted', n_colors=20)
            sns.set_style({'font.family': 'serif', 'font.serif': ['SimHei']})
            kwargs['palette'] = itertools.cycle(sns.color_palette('muted'))
            return func(*args, **kwargs)

    return maxent_style_inner


def remove_palette(func):
    def remove_palette_inner(*args, **kwargs):
        print(func.__name__)
        if kwargs.get('palette') is not None:
            kwargs.pop('palette')
        return func(*args, **kwargs)

    return remove_palette_inner
