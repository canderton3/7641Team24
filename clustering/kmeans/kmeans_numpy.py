from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import numpy as np
import matplotlib.pyplot as plt
import imageio


# load image to recolor #
img_file = ''' '''


# helper functions #

def pairwise_dist(x, y):
    dist = x[:, :, None] - y[:, :, None].T
    dist = np.sqrt((dist * dist).sum(axis=1))
    return dist

def softmax(logits):
    top = np.exp(logits - logits.max(axis=1, keepdims=True))
    bottom = np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1, keepdims=True)
    return top / bottom


def logsumexp(logits):
    one = logits.max(axis=1, keepdims=True)
    two = np.log(np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1, keepdims=True))
    return one + two


def plot_images(img_list, title_list, figsize=(11, 6)):
    assert len(img_list) == len(title_list)
    fig, axes = plt.subplots(1, len(title_list), figsize=figsize)
    for i, ax in enumerate(axes):
        ax.imshow(img_list[i] / 255.0)
        ax.set_title(title_list[i])
        ax.axis('off')

def plot_scatter(samples, ids):
    colors = np.zeros((len(ids), 3))
    choices = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    num_points = []
    for i in range(3):
        num_points.append(np.sum(ids == i))
    maps = np.argsort(num_points)
    for i in range(3):
        colors[np.where(ids == maps[i]), :] = choices[i]
    plt.scatter(samples[:, 0], samples[:, 1], s=1, color=colors)
    plt.axis('equal')


# KMeans clustering #
class KMeans(object):
    def __init__(self):
        pass

    def _init_centers(self, points, k, **kwargs):
        k = np.minimum(K, (np.unique(points)).size)
        n, d = points.shape
        maximum = np.max(points, axis=0)
        return maximum * np.random.rand(k, d)

    def _update_assignment(self, centers, points):
        cluster_idx = np.argmin(pairwise_dist(points, centers), axis=1)
        return np.transpose(cluster_idx)

    def _update_centers(self, old_centers, cluster_idx, points):
        k, d = old_centers.shape
        centers = np.empty((k, d))
        keep = []

        for i in range(k):
            cluster = points[np.argwhere(cluster_idx == i)]
            if cluster.shape[0] > 0:
                centers[i] = np.sum(cluster, axis=0) / cluster.shape[0]
                keep.append(centers[i])
        new_centers = np.array(keep)

        return new_centers

    def _get_loss(self, centers, cluster_idx, points):
        cluster_idx = self._update_assignment(centers, points)
        return (np.linalg.norm(points - centers[cluster_idx]) ** 2).sum()

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            k = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
        return cluster_idx, centers


# load image, plot original and clustered image #
image = imageio.imread(img_file)
im_height, im_width, im_channel = image.shape

flat_img = np.reshape(image, [-1, im_channel]).astype(np.float32)

cluster_ids, centers = KMeans()(flat_img, K=5)

kmeans_img = np.reshape(centers[cluster_ids], (im_height, im_width, im_channel))

plot_images([image, kmeans_img], ['origin', 'kmeans'])
