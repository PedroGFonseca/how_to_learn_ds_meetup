import pandas as pd
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.datasets import make_moons, make_circles, make_classification

import graphviz
from mlxtend.plotting import plot_decision_regions


def _get_scikit_datasets(n_points=100):
    rng = np.random.RandomState(2)
    X, y = make_classification(n_samples=n_points, n_features=2,
                               n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)

    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    datasets = [make_moons(n_samples=n_points, noise=0.3, random_state=0),
                make_circles(n_samples=n_points, noise=0.2, factor=0.5,
                             random_state=1),
                linearly_separable
                ]
    return datasets


def get_importances(model, feature_names):
    dict(zip(feature_names, model.feature_importances))
    return pd.Series()


def get_toy_data():
    toy_data = {
        'a': [1, 2, 2],
        'b': ['cat', 'cat', 'dog'],
        'c': ['yes', 'yes', 'no']
    }

    return pd.DataFrame(toy_data)


def get_circle(n_points=100):
    datasets = _get_scikit_datasets(n_points)
    i = 1
    q = pd.DataFrame(datasets[i][0])
    q['c'] = datasets[i][1]
    q.columns = ['a', 'b', 'c']
    return q


def get_ying_yang(n_points=100):
    datasets = _get_scikit_datasets(n_points)
    i = 0
    q = pd.DataFrame(datasets[i][0])
    q['c'] = datasets[i][1]
    q.columns = ['a', 'b', 'c']
    return q


def get_linear_separable(n_points=100):
    datasets = _get_scikit_datasets(n_points)
    i = 2
    q = pd.DataFrame(datasets[i][0])
    q['c'] = datasets[i][1]
    q.columns = ['a', 'b', 'c']
    return q


def get_cross_data(n_points=10):
    x = np.linspace(1, n_points, n_points)
    y = np.linspace(1, n_points * 100, n_points)
    q = pd.DataFrame(list(product(x, y)), columns=['a', 'b'])
    q['c'] = 0
    q = q.fillna(0)
    q.loc[((q['b'] < q['b'].median()) & (q['a'] < q['a'].median())), 'c'] = 1
    q.loc[((q['b'] > q['b'].median()) & (q['a'] > q['a'].median())), 'c'] = 1
    return q


def scatterplot(df, feature_1, feature_2, target, small=False):
    if small:
        size=(3, 3)
    else:
        size=(7, 7)
    fig, ax = plt.subplots(figsize=size)
    for name, group in df.groupby(target):
        ax.plot(group[feature_1], group[feature_2], marker='+', linestyle='', ms=10, label=name)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.legend()
    plt.title('Scatterplot of %s vs %s, with %s as the color' % (feature_1, feature_2, target))
    plt.show()


def plot_boundaries(X, y, clf):
    features = X.columns
    plot_decision_regions(X=X.values,
                      y=y.values,
                      clf=clf,
                      res=0.02,
                      legend=2)
    # Adding axes annotations
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Predictions of the model')
    plt.show()


def plot_tree(tree_classifier, X_train, class_names):
    dot_data = tree.export_graphviz(tree_classifier, out_file=None,
                         feature_names=X_train.columns,
                         class_names=class_names,
                         filled=True, rounded=True,
                         special_characters=True)

    return graphviz.Source(dot_data)