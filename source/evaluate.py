import numpy as np
from math import sqrt


def distance(x, y):
    """Euclidean distance between two samples"""
    return sqrt(np.sum(np.power(x - y, 2)))


def clusters_dissimilarity(c1, c2):
    """Calculate dissimilarity metric between two clusters"""
    min_dist = None
    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            dist = distance(c1[i], c2[j])
            if not min_dist or dist < min_dist:
                min_dist = dist
    return min_dist


def cluster_diameter(cluster):
    """Compute the diameter of a cluster"""
    max_dist = -1
    for i in range(cluster.shape[0]):
        for j in range(cluster.shape[0]):
            if i != j:
                dist = distance(cluster[i], cluster[j])
                if dist > max_dist:
                    max_dist = dist
    return max_dist


def dunn_index(clusters):
    """Compue dunn index from cluster datapoints"""
    max_diameter = -1
    min_ratio = None
    for c in clusters:
        print("diameter")
        diameter = cluster_diameter(c)
        if diameter > max_diameter:
            max_diameter = diameter
    k = len(clusters)
    for i in range(0, k):
        for j in range(i + 1, k):
            print("ratio")
            dissimilarity = clusters_dissimilarity(clusters[i], clusters[j])
            ratio = dissimilarity / max_diameter
            if not min_ratio or ratio < min_ratio:
                min_ratio = ratio
    return min_ratio


