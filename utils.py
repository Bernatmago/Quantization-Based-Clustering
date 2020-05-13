import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from os.path import join
import pandas as pd
import cv2
plt.style.use('ggplot')


def load_data(dataset, n_dims=-1, has_labels=True):
    data_dir = 'data'
    df = pd.read_csv(join(data_dir, dataset + '.csv'))
    if has_labels:
        x = df.to_numpy()[:, :-1]
        y = df.to_numpy()[:, -1]
    else:
        x = df.to_numpy()
        y = None
    if 0 < n_dims < x.shape[1]:
        pca = PCA(n_components=n_dims)
        x = pca.fit_transform(x)
    return x, y


def image_as_dataset(filename, im_height=150):
    data_dir = 'data'
    im = cv2.imread(join(data_dir, filename + '.jpg'))
    if im_height:
        y, x, z = im.shape
        ratio = im_height / float(y)
        dim = (int(x * ratio), im_height)
        im = cv2.resize(im, dim, interpolation=cv2.INTER_CUBIC)
    y, x, z = im.shape
    x_idx, y_idx = np.unravel_index(np.arange(x*y), im.shape[:2])
    im2d = im.reshape(x*y, z)
    im2d = np.hstack((im2d, x_idx.reshape(-1, 1), y_idx.reshape(-1, 1)))
    return im2d, im


def cluster_list(X, y):
    n = np.unique(y)
    return [X[y == i, :] for i in n]


def plot_comparison(x1, y1, x2, y2, alg1, alg2, dataset):
    plot_dir = 'output'
    if x1.shape[1] > 2:
        p = PCA(n_components=2)
        x1 = p.fit_transform(x1)
        x2 = p.transform(x2)
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(alg1)
    axs[0].scatter(x1[:, 0], x1[:, 1], c=y1)
    axs[1].set_title(alg2)
    axs[1].scatter(x2[:, 0], x2[:, 1], c=y2)
    plt.savefig(join(plot_dir, '{}_comp.png'.format(dataset)))



def plot_segmented(x1, y1, x2, y2, c1, c2, shape, algs, dataset):
    plot_dir = 'output'
    fig, axs = plt.subplots(1, 2)

    colours = np.array([
        [0, 0, 255], #blue
        [204, 0, 0], #red
        [0, 153, 51], #green
        [53, 0, 153] #purple
    ])
    for n, (d, l, c) in enumerate(zip([x1, x2], [y1, y2], [c1, c2])):
        im = np.zeros(shape, dtype=int)
        seg = np.zeros(shape, dtype=int)

        for s, i in zip(d, l):
            im[int(s[-2]), int(s[-1]), :] = s[:3]

            seg[int(s[-2]), int(s[-1]), :] = colours[i, :]
        axs[n].grid(False)
        axs[n].imshow(im)
        axs[n].imshow(seg, alpha=0.35)
        axs[n].set_xticks([])
        axs[n].set_yticks([])

        axs[n].set_title(algs[n])
    plt.tight_layout()
    plt.savefig(join(plot_dir, '{}_seg.png'.format(dataset)))
    pass


def plot_metrics(results, labels, categories, metric_name, params_name, dataset):
    plot_dir = 'output'
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
    ax.set_title('{}: {} by {}'.format(dataset, metric_name, params_name))
    plt.tight_layout()
    plt.savefig(join(plot_dir, '{}_{}_{}.png'.format(dataset, metric_name, params_name)))

