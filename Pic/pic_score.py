# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/11/9
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import numpy as np
from Pic.maxent_style import maxent_style
"""
this file is sustract from mlxtend which do not support for adjust matplotlib style your self;
cause we want to use seaborn to get better picture and save to where we want
"""


@maxent_style
def plot_learning_curves(X_train, y_train,
                         X_test, y_test,
                         clf,
                         ax,
                         palette=None,
                         train_marker='o',
                         test_marker='^',
                         scoring='misclassification error',
                         legend_loc='best'):
    """Plots learning curves of a classifier.

    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
        Feature matrix of the training dataset.
    y_train : array-like, shape = [n_samples]
        True class labels of the training dataset.
    X_test : array-like, shape = [n_samples, n_features]
        Feature matrix of the test dataset.
    y_test : array-like, shape = [n_samples]
        True class labels of the test dataset.
    clf : Classifier object. Must have a .predict .fit method.
    train_marker : str (default: 'o')
        Marker for the training set line plot.
    test_marker : str (default: '^')
        Marker for the test set line plot.
    scoring : str (default: 'misclassification error')
        If not 'misclassification error', accepts the following metrics
        (from scikit-learn):
        {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
        'f1_weighted', 'f1_samples', 'log_loss',
        'precision', 'recall', 'roc_auc',
        'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
        'median_absolute_error', 'r2'}
    suppress_plot=False : bool (default: False)
        Suppress matplotlib plots if True. Recommended
        for testing purposes.
    print_model : bool (default: True)
        Print model parameters in plot title if True.
    style : str (default: 'fivethirtyeight')
        Matplotlib style
    legend_loc : str (default: 'best')
        Where to place the plot legend:
        {'best', 'upper left', 'upper right', 'lower left', 'lower right'}

    Returns
    ---------
    errors : (training_error, test_error): tuple of lists

    """
    if scoring != 'misclassification error':
        from sklearn import metrics

        scoring_func = {
            'accuracy': metrics.accuracy_score,
            'average_precision': metrics.average_precision_score,
            'f1': metrics.f1_score,
            'f1_micro': metrics.f1_score,
            'f1_macro': metrics.f1_score,
            'f1_weighted': metrics.f1_score,
            'f1_samples': metrics.f1_score,
            'log_loss': metrics.log_loss,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'roc_auc': metrics.roc_auc_score,
            'adjusted_rand_score': metrics.adjusted_rand_score,
            'mean_absolute_error': metrics.mean_absolute_error,
            'mean_squared_error': metrics.mean_squared_error,
            'median_absolute_error': metrics.median_absolute_error,
            'r2': metrics.r2_score}

        if scoring not in scoring_func.keys():
            raise AttributeError('scoring must be in', scoring_func.keys())

    else:
        def misclf_err(y_predict, y):
            return (y_predict != y).sum() / float(len(y))

        scoring_func = {
            'misclassification error': misclf_err}

    training_errors = []
    test_errors = []

    rng = [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:]
    for r in rng:
        model = clf.fit(X_train[:r], y_train[:r])

        y_train_predict = clf.predict(X_train[:r])
        y_test_predict = clf.predict(X_test)

        train_misclf = scoring_func[scoring](y_train[:r], y_train_predict)
        training_errors.append(train_misclf)

        test_misclf = scoring_func[scoring](y_test, y_test_predict)
        test_errors.append(test_misclf)

    ax.plot(np.arange(10, 101, 10), training_errors, c=next(palette),
             label='training set', marker=train_marker)
    ax.plot(np.arange(10, 101, 10), test_errors, c=next(palette),
             label='test set', marker=test_marker)
    ax.set_xlabel('Training set size in percent')
    ax.set_ylabel('Performance ({})'.format(scoring))
    ax.set_title('Learning Curves For {}'.format(model.__class__.__name__))
    ax.legend(loc=legend_loc, numpoints=1)
    ax.set_xlim([0, 110])
    max_y = max(max(test_errors), max(training_errors))
    min_y = min(min(test_errors), min(training_errors))
    ax.set_ylim([min_y - min_y * 0.15, max_y + max_y * 0.15])
    errors = (training_errors, test_errors)
    return ax, errors