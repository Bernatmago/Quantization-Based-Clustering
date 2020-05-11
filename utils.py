import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from os.path import join
import pandas as pd
plt.style.use('ggplot')


def load_data(dataset, n_dims=-1):
    data_dir = 'data'
    df = pd.read_csv(join(data_dir, dataset + '.csv'))
    x = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]
    if n_dims > 0:
        pca = PCA(n_components=n_dims)
        x = pca.fit_transform(x)
    return x, y


def cluster_list(X, y):
    n = np.unique(y)
    return [X[y == i, :] for i in n]


def plot_comparison(x1, y1, x2, y2, alg1, alg2):
    if x1.shape[1] > 2:
        p = PCA(n_components=2)
        x1 = p.fit_transform(x1)
        x2 = p.transform(x2)
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(alg1)
    axs[0].scatter(x1[:, 0], x1[:, 1], c=y1)
    axs[1].set_title(alg2)
    axs[1].scatter(x2[:, 0], x2[:, 1], c=y2)
    plt.show()
    plt.savefig('test.png')


def plot_metrics(results, labels, categories, metric_name, params_name):
    gap = 0.35
    x = np.arange(len(results[0]))
    fig, ax = plt.subplots()
    for i, r in enumerate(results):
        ax.bar(x + i * gap, r, width=gap, label=categories[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_name)
    ax.set_xlabel(params_name)
    ax.legend(loc='best')
    ax.set_title('{} by {}'.format(metric_name, params_name))
    plt.show()
    plt.savefig('test2.png')
