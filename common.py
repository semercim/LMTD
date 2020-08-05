import numpy as np
from numpy.core.multiarray import ndarray
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import pairwise_distances


def knn_classify_many(training_set, training_labels, test_set, test_labels, num_neighbors, run_training=False):
    clf = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean')
    clf.fit(training_set, training_labels)
    input_test_predictions = clf.predict(test_set)
    test_result = np.sum(input_test_predictions ==
                         test_labels) * 100.0 / float(len(test_labels))
    if not run_training:
        return 0.0, test_result

    loo = LeaveOneOut()
    training_set = np.array(training_set)
    input_training_predictions = []
    for train_index, test_index in loo.split(training_set):
        clf = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
        clf.fit(training_set[train_index], training_labels[train_index])
        input_training_predictions.append((clf.predict(training_set[test_index]) == training_labels[test_index]).astype(int))
    training_result = (np.sum(input_training_predictions) * 100.0 / float(len(training_set)))

    return training_result, test_result


def knn_classify0(training_set, training_labels, test_set, test_labels, num_neighbors):
    clf = NearestCentroid(metric='euclidean')
    clf.fit(training_set, training_labels)
    input_test_predictions = clf.predict(test_set)
    test_result = np.sum(input_test_predictions ==
                         test_labels) * 100.0 / float(len(test_labels))  # type: ndarray
    return 0.0, test_result


def sort_neighbors(training_set, training_labels, test_set):
    distances = pairwise_distances(test_set, training_set, metric="euclidean")
    indices = np.argsort(distances, axis=1)
    sorted_neighbors = []
    sorted_distances = []
    for i in range(0, test_set.shape[0]):
        sorted_neighbors.append([training_labels[j] for j in indices[int(i), :]])
        sorted_distances.append([distances[i, j] for j in indices[int(i), :]])
    return np.array(sorted_neighbors), np.array(sorted_distances), np.array(indices)


def average_precision_at_k(sorted_neighbors, test_labels, k=None):
    matched_labels = []
    for i in range(0, len(test_labels)):
        matched_labels.append(sorted_neighbors[i, :] == test_labels[i])
    matched_labels = np.array(matched_labels)
    precisions = np.zeros(matched_labels.shape)
    if k is not None:
        for K in range(k):
            precisions[:, K] = (np.sum(matched_labels[:, :(K+1)], axis=1)*(sorted_neighbors[:, K] == test_labels))/float(K+1)
    else:
        for K in range(sorted_neighbors.shape[1]):
            precisions[:, K] = (np.sum(matched_labels[:, :(K+1)], axis=1)*(sorted_neighbors[:, K] == test_labels))/float(K+1)
    average_precisions = np.sum(precisions, axis=1) / np.sum(precisions > 0.0, axis=1)

    return matched_labels, precisions, average_precisions


def mean_average_precision(training_set, training_labels, test_set, test_labels):
    sorted_neighbors, sorted_distances, indices = sort_neighbors(training_set, training_labels, test_set)
    matched_labels, precisions, average_precisions = average_precision_at_k(sorted_neighbors, test_labels, k=None)
    return np.mean(average_precisions)
