import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy import sparse
from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian as graph_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.manifold.spectral_embedding_ import _set_diag
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score


def plotData(data, labels, cores_indices=[], title="DBSCAN", plotShow=True):
    """
    Plot data
    :param data: data matrix (n_sample x 2)
    :param labels: array of labels (-1 is noise)
    :param cores_indices: index of core point
    :param title: the figure's title
    :param plotShow: show figure if plotShow
    :param fileName: Name of file in case of saving figure. Default DBSCAN.png
    :return:
    """

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    core_samples_mask = np.zeros_like(labels, dtype=bool)

    core_samples_mask[cores_indices] = True

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title(title)
    if plotShow:
        plt.show()
    plt.close()


def predict_k(affinity_matrix):
    """
    Predict number of clusters based on the eigengap.
    Parameters
    ----------
    affinity_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix.
        Each element of this matrix contains a measure of similarity between two of the data points.
    Returns
    ----------
    k : integer
        estimated number of cluster.
    Note
    ---------
    If graph is not fully connected, zero component as single cluster.
    References
    ----------
    A Tutorial on Spectral Clustering, 2007
        Luxburg, Ulrike
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """

    """
    If normed=True, L = D^(-1/2) * (D - A) * D^(-1/2) else L = D - A.
    normed=True is recommended.
    """
    normed_laplacian, dd = graph_laplacian(affinity_matrix, normed=True, return_diag=True)

    laplacian = _set_diag(normed_laplacian, 1, norm_laplacian=0)

    """
    n_components size is N - 1.
    Setting N - 1 may lead to slow execution time...
    """
    n_components = affinity_matrix.shape[0] - 1

    """
    shift-invert mode
    The shift-invert mode provides more than just a fast way to obtain a few small eigenvalues.
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    The normalized Laplacian has eigenvalues between 0 and 2.
    I - L has eigenvalues between -1 and 1.
    """
    eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues = -eigenvalues[::-1]  # Reverse and sign inversion.

    max_gap = 0
    gap_pre_index = 0
    for i in range(1, eigenvalues.size):
        gap = eigenvalues[i] - eigenvalues[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_pre_index = i - 1

    k = gap_pre_index + 1

    return k


def get_means_std(data):
    means = np.mean(data, axis=0)
    means = np.expand_dims(means, axis=1)
    stds = np.expand_dims(np.std(data, axis=0), axis=1)
    mean_stds = np.concatenate([means, stds], axis=1)
    return mean_stds


def flows_k_means(data, max_k=5):

    mean_stds = get_means_std(data)

    silhou_score = []
    esimators = []
    for ncluster in range(2, max_k+1, 1):
        esimators.append(KMeans(init='k-means++', n_clusters=ncluster, n_init=10))
        esimators[ncluster - 2].fit(mean_stds)
        silhou_score.append(silhouette_score(mean_stds, esimators[ncluster - 2].labels_, metric='euclidean'))

    silhou_score = np.array(silhou_score)
    n_clusters = np.argmax(silhou_score) + 2
    k_means = esimators[n_clusters - 2]
    # plt.title('Silhouette Score')
    # plt.plot(range(2, 5, 1), silhou_score.T)
    # plt.show()
    # plt.close()

    # plotData(mean_stds, labels=k_means.labels_, title='K-Means', plotShow=False)
    return k_means, n_clusters


def flows_DBSCAN(data, eps=0.2, min_samples=5):

    means_stds = get_means_std(data)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(means_stds)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Consider noise as a label.
    n_clusters_ = len(set(labels))

    # plotData(data=means_stds, labels=db.labels_, cores_indices=db.core_sample_indices_,
    #          plotShow=False)

    return db, n_clusters_


