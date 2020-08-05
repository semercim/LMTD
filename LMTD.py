# -*- coding: utf-8 -*-
"""
@author: semercim (murat.semerci@boun.edu.tr)
"""

import copy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import LeaveOneOut
from sklearn import neighbors
from sklearn.base import BaseEstimator, TransformerMixin
import sktensor
# from sklearn.preprocessing import normalize
from scipy import linalg as spla


class LMTD(BaseEstimator, TransformerMixin):
    """
    This class implements LMTD
    """
    def __init__(self, num_components=None, max_iter=100, dimensions=None,
                 mu=1.0, beta=1.0, gamma=1.0, alpha=None, margin=None,
                 learning_rate=0.01, num_neighbors=1, update_iter=1, verbose=False,
                 initialization='randomized', result_set='min_loss',
                 init_reduced_dim=True):
        """
        @author: semercim
        :param num_components: a vector of N-dim, the reduced dimensions in each of N-ways
        :param max_iter: the number of iterations, default 100
        :param dimensions: the input dimensions
        :param mu: the reconstruction penalty, default 1.0
        :param beta: the target distance penalty, default 1.0
        :param gamma: the impostor distance penalty, default 1.0
        :param alpha: the soft orthogonality penalty
        :param margin: the margin value
        :param learning_rate: the constant learning rate, default 0.01
        :param num_neighbors: the number of neighbor, k in k-nn
        :param update_iter: the iteration interval for checking the impostors, default 1
        :param verbose: print info messages, default False
        :param initialization: the way initial projection matrices are set, default 'randomized'
        :param result_set: the returned optimized projection set, default 'min_loss'
        :param init_reduced_dim: initialize the projections in reduced dimensions, default True
        """
        self.max_iter = max_iter
        self.num_components = num_components
        self.num_neighbors = num_neighbors
        self.verbose = verbose
        self.gamma = gamma
        self.dimensions = dimensions
        self.num_ways = 0
        # the instance itself is also its own neighbor
        self.num_neighbors = num_neighbors + 1
        if dimensions:
            self.num_ways = len(self.dimensions)
        self.projections = []  # => Ui's
        # min loss results
        self.min_impostors_projections = []
        self.min_impostors_loss = np.Inf
        self.min_impostors_impostors = np.Inf
        # min loss results
        self.min_loss_projections = []
        self.min_loss_loss = np.Inf
        self.min_loss_impostors = np.Inf
        # min impostor min loss pair results
        self.min_pair_projections = []
        self.min_pair_loss = np.Inf
        self.min_pair_impostors = np.Inf
        self.X_ = None
        self.y_ = []
        self.classes_ = []
        self.n_classes_ = 0
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.margin = margin
        self.learning_rate = learning_rate
        self.update_iter = update_iter
        # set initialization
        self.initialization = 'randomized'
        if initialization not in ['mda', 'randomized', 'Tucker']:
            print("The initialization is not one of 'mda', 'randomized', 'Tucker' ")
            print("Setting 'randomized' as the default initialization")
        else:
            self.initialization = initialization
        # set result set
        self.result_set = 'min_loss'
        if result_set not in ['min_loss', 'min_pair', 'min_impostors']:
            print("The initialization is not one of 'min_loss', 'min_pair', 'min_impostors' ")
            print("Setting 'min_loss' as the default result_set")
        else:
            self.result_set = result_set
        self._name = 'LMTD'
        self.unfolded = []
        self.init_reduced_dim = init_reduced_dim

    def initialize_projection_matrices(self, tensors_list, labels=None):
        """
        @author: semercim
        :param tensors_list: the input tensors
        :param labels: the labels
        :return:
            us: the list of projection matrices

        This function is to be used initializing the core tensors and projection
        tensor for the first time.
        The projections matrices are "input_dim X reduced_dim"
        """
        # Get the number of instances and dimension
        num_instances = len(tensors_list)

        if self.initialization == 'randomized':
            # create a random input_dim X input_dim matrix
            # then select input_dim X reduced_dim unit row
            for mode in range(0, self.num_ways):
                random_matrix = np.random.random((tensors_list[0].shape[mode],
                                                 tensors_list[0].shape[mode]))
                random_matrix, _ = np.linalg.qr(random_matrix)
                if self.init_reduced_dim:
                    # take initial reduced_dims columns and transpose it.
                    self.projections.append(random_matrix[:self.num_components[mode], :].T)
                else:
                    self.projections.append(random_matrix.T)
            return

        if self.initialization == 'mda':
            for dimension in self.dimensions:
                self.projections.append(np.eye(dimension))

            _, member_counts = np.unique(labels, return_counts=True)
            for dimension in range(self.num_ways):
                projected = copy.deepcopy(tensors_list)  # =>Yi's
                projection_list_inds = list(range(0, self.num_ways))
                projection_list_inds.remove(dimension)
                tmp_projections = tuple([self.projections[i] for i in projection_list_inds])
                for index in range(num_instances):
                    projected[index] = projected[index].ttm(tmp_projections,
                                                            dimension,
                                                            without=True)
                    projected[index] = projected[index].unfold(dimension)
                # calculate the projected tensor overall mean and the class means
                # it changes in each iteration
                class_means = []
                overall_mean = np.mean(projected, axis=0)  # Overall Mean
                for class_ in range(self.n_classes_):  # Class Mean
                    class_means.append(np.mean(np.array(projected)
                                               [labels == self.classes_[class_], :],
                                               axis=0))
                # calculate the between and within class spread
                class_between_spread = 0.0
                class_within_spread = 0.0
                for class_ in range(self.n_classes_):
                    current_class_difference = (class_means[class_] - overall_mean)
                    class_between_spread += (member_counts[class_] *
                                             np.dot(current_class_difference,
                                                    current_class_difference.T))
                    for index in range(num_instances):
                        current_instance_difference = ((self.classes_[class_] == labels[index]) *
                                                       (projected[index] - class_means[class_]))
                        class_within_spread += np.dot(current_instance_difference,
                                                      current_instance_difference.T)
                # take first the predetermined eigenvectors
                eigen_values, eigen_vectors = spla.eig(class_between_spread,
                                                       b=class_within_spread)
                idx = eigen_values.argsort()[::-1]
                eigen_vectors = eigen_vectors[:, idx]
                self.projections[dimension] = eigen_vectors
                self.projections[dimension] /= np.linalg.norm(self.projections[dimension], axis=0)

            for dimension in range(self.num_ways):
                if self.init_reduced_dim:
                    self.projections[dimension] = self.projections[dimension][:self.num_components[dimension], :].T
                else:
                    self.projections[dimension] = self.projections[dimension].T
            return

        # initial individual tucker decompositions for each tensor
        projection_holder = []
        for tensor in tensors_list:
            _, projection = sktensor.tucker.hooi(tensor, self.num_components, init='nvecs')
            projection_holder.append(projection)

        # the orthonormal projection matrices are calculated from averaged projections
        for mode in range(0, self.num_ways):
            u_t = 0.0
            for index in range(0, num_instances):
                u_t += projection_holder[index][mode]
            averaged_u_t = u_t / num_instances
            # get the orthonormal projections using svd over averaged projections
            uh, _, _ = np.linalg.svd(averaged_u_t)
            if self.init_reduced_dim:
                self.projections.append(uh[:self.num_components[mode], :].T)
            else:
                self.projections.append(uh.T)
        return

    def select_target_neighbors(self, tensors_list):
        """
        @author: semercim
        :param tensors_list: the list of the N input tensors
        :return:
            target_neighbors: a numpy array of Nxk, k neighbors for each instances
            (ith row for ith instance)
        """
        num_instances = len(tensors_list)
        target_neighbors = np.zeros((num_instances, self.num_neighbors), dtype=int)

        # vectorize the tensors
        vectorized_tensors = []
        for tensor in tensors_list:
            vectorized_tensors.append(copy.deepcopy(tensor).flatten('F'))

        # for each class, find the targets for that class' instances
        for class_ in self.classes_:
            class_ind, = np.where(np.equal(self.y_, class_))
            dist = euclidean_distances(np.array(vectorized_tensors)[class_ind], squared=True)
            # exclude the current instance as its own target
            np.fill_diagonal(dist, np.inf)
            neigh_ind = np.argpartition(dist, self.num_neighbors - 1, axis=1)
            neigh_ind = neigh_ind[:, :self.num_neighbors]
            # we sort again but only the k neighbors
            row_ind = np.arange(len(class_ind))[:, None]
            neigh_ind = neigh_ind[row_ind, np.argsort(dist[row_ind, neigh_ind])]
            target_neighbors[class_ind] = class_ind[neigh_ind]

        return target_neighbors

    def find_impostors(self, vectorized_tensors, index, target_neighbors):
        """
        @author: semercim
        :param vectorized_tensors:
        :param index:
        :param target_neighbors:
        :return:
            violated:
            dist_tn:
            distances:
        """
        dist_tn = np.zeros((1, target_neighbors.shape[1]))
        # for the current instance (index), find its target distances
        for k in range(0, target_neighbors.shape[1]):
            dist_tn[0, k] = (np.sum(np.square(vectorized_tensors[index, :] -
                                              vectorized_tensors[target_neighbors[index, k], :])) +
                             self.margin)

        # find the maximum target distance
        target_radius = np.max(dist_tn)
        distances = np.sum(np.square(vectorized_tensors -
                                     vectorized_tensors[index]), axis=1)
        distances[index] = np.Inf
        # find the impostors whose distance is smaller than
        # the max target distance
        violated = np.where(np.logical_and((distances <= target_radius),
                                           (self.y_ != self.y_[index])))

        return violated, dist_tn, distances

    def calculate_loss(self, tensors_list, core_tensors, target_neighbors):
        """
        @author: semercim
        :param tensors_list: list of input tensors (N instances)
        :param core_tensors: the core tensors (N instances)
        :param target_neighbors: the target neighbors (N x k)
        :return:
            loss: the value of loss function
        """
        pass

    def get_intermediate_variable(self):
        """
        @author: semercim
        :return:
            Ts:
            Ss:
        """
        Ts = []  # Ts in the journal, projection X projection.T
        Ss = []  # For each way, the other Ts are multiplied in reversed index
        for mode in range(len(self.projections)):
            Ts.append(np.dot(self.projections[mode], self.projections[mode].T))
        for projection in range(self.num_ways):
            temp = 1.0
            for mode in reversed(range(self.num_ways)):
                if projection == mode:
                    continue
                temp = np.kron(temp, Ts[mode])
            Ss.append(temp)
        return Ts, Ss

    def update_projections(self, tensors_list, target_neighbors):
        """
        @author: semercim
        :param tensors_list: list of input tensors (N instances)
        :param target_neighbors: the target neighbors indices (N x k)
        :return:
            us: the list of projection matrices
        """
        # Get the number of instances and dimension

        pass

    def get_core_tensors(self, tensors_list):
        """
        @author: semercim
        :param tensors_list:
        :return:
        """
        # Get the number of instances and dimension

        # initial individual tucker decompositions for each tensor
        core_tensors = []

        # calculate the core tensors with averaged orthonormal projections
        for tensor in tensors_list:
            core_tensor = copy.deepcopy(tensor)
            for mode in range(self.num_ways):
                # in our
                core_tensor = core_tensor.ttm(self.projections[mode].T, mode)
            core_tensors.append(core_tensor)

        # the core tensors and projections are returned
        return core_tensors

    def fit_tensor(self, tensors_list, labels, max_iter=None):
        """
        @author: semercim
        :param tensors_list:
        :param labels:
        :param max_iter:
        :return:
        """
        if self.margin is None:
            self.margin = np.prod(tensors_list[0].shape)

        if max_iter is not None:
            self.max_iter = max_iter
        if not self.dimensions:
            self.dimensions = tensors_list[0].shape
        if not self.num_ways:
            self.num_ways = len(self.dimensions)
        self.y_ = labels

        classes, _ = np.unique(labels, return_counts=True)
        self.classes_ = classes
        self.n_classes_ = len(self.classes_)

        losses = []

        if self.alpha is None:
            self.alpha = np.prod(tensors_list[0].shape)

        # initialize unfolded tensors
        # TODO: check the data size before unfolding it for memory usage
        num_instances = len(tensors_list)
        for mode in range(self.num_ways):
            self.unfolded.append([])
            for index in range(num_instances):
                self.unfolded[mode].append(tensors_list[index].unfold(mode))

        # At first, we need to initialize the core tensors and projection matrices
        self.initialize_projection_matrices(tensors_list, labels)
        core_tensors = self.get_core_tensors(tensors_list)
        # Decide the target neighbors, they will never be changed later!
        target_neighbors = self.select_target_neighbors(tensors_list)
        # calculate the loss for the first time.
        loss, num_total_impostors = self.calculate_loss(tensors_list, core_tensors, target_neighbors)
        print("\n" + self._name + " Initial loss: " + str(loss) + "\n")
        min_loss_it = 0
        min_pair_it = 0
        min_impostors_it = 0
        # store the old projections for book-keeping
        self.min_pair_projections = copy.deepcopy(self.projections)
        self.min_loss_projections = copy.deepcopy(self.projections)
        self.min_impostors_projections = copy.deepcopy(self.projections)
        if loss > 0.0:
            for iteration in range(self.max_iter):
                self.update_projections(tensors_list, target_neighbors)
                core_tensors = self.get_core_tensors(tensors_list)
                loss, num_total_impostors = self.calculate_loss(tensors_list, core_tensors, target_neighbors)
                print(self._name + " iteration=" + str(iteration) + " loss=" + str(loss))
                losses.append(loss)
                if self.min_loss_loss >= loss:
                    self.min_loss_loss = loss
                    self.min_loss_impostors = num_total_impostors
                    self.min_loss_projections = copy.deepcopy(self.projections)
                    min_loss_it = iteration

                if self.min_pair_loss >= loss and self.min_pair_impostors >= num_total_impostors:
                    self.min_pair_loss = loss
                    self.min_pair_impostors = num_total_impostors
                    self.min_pair_projections = copy.deepcopy(self.projections)
                    min_pair_it = iteration

                if self.min_impostors_impostors >= num_total_impostors:
                    self.min_impostors_loss = loss
                    self.min_impostors_impostors = num_total_impostors
                    self.min_impostors_projections = copy.deepcopy(self.projections)
                    min_impostors_it = iteration
                if loss < 0.1:
                    break

        for mode in range(self.num_ways):
            self.min_loss_projections[mode] = self.min_loss_projections[mode][:, :self.num_components[mode]]
            self.min_pair_projections[mode] = self.min_pair_projections[mode][:, :self.num_components[mode]]
            self.min_impostors_projections[mode] = self.min_impostors_projections[mode][:, :self.num_components[mode]]

        if self.verbose:
            print("\n" + self._name + " Min Loss Loss :", str(self.min_loss_loss), " Impostors: ",
                  str(self.min_loss_impostors), " Iteration :",
                  str(min_loss_it), "\n")

            print("\n" + self._name + " Min Pair Loss :", str(self.min_pair_loss), " Impostors: ",
                  str(self.min_pair_impostors), " Iteration :",
                  str(min_pair_it), "\n")

            print("\n" + self._name + " Min Impostors Loss :", str(self.min_impostors_loss), " Impostors: ",
                  str(self.min_impostors_impostors), " Iteration :",
                  str(min_impostors_it), "\n")

        if self.result_set == 'min_loss':
            self.projections = copy.deepcopy(self.min_loss_projections)

        if self.result_set == 'min_pair':
            self.projections = copy.deepcopy(self.min_pair_projections)

        if self.result_set == 'min_impostors':
            self.projections = copy.deepcopy(self.min_impostors_projections)

        self.X_ = self.get_projections(tensors_list)
        return self

    def fit(self, instances, labels, sample_weight=None):
        """
        @author: semercim
        :param instances:
        :param labels:
        :param sample_weight:
        :return:
        """
        if not self.num_ways:
            raise ValueError("The dimensions must be set before fitting the model!")
        if not self.dimensions:
            raise ValueError("The dimensions must be set before fitting the model!")

        num_of_instances = instances.shape[0]
        tensor_list = []
        for index in range(num_of_instances):
            tensor_list.append(sktensor.dtensor(instances[index, :].reshape(self.dimensions)))
        self.fit_tensor(tensor_list, labels)

    def predict_(self, projected_x):
        """
        @author: semercim
        :param projected_x:
        :return:
        """
        if np.array_equal(self.X_, projected_x):
            loo = LeaveOneOut()
            labels = np.zeros((self.X_.shape[0], 1))
            for train_index, test_index in loo.split(self.X_):
                clf = neighbors.KNeighborsClassifier(n_neighbors=self.num_neighbors)
                clf.fit(self.X_[train_index], self.y_[train_index])
                labels[test_index] = clf.predict(self.X_[test_index])
            return labels.flatten()
        clf = neighbors.KNeighborsClassifier(n_neighbors=self.num_neighbors)
        clf.fit(self.X_, self.y_)
        return np.array(clf.predict(projected_x)).flatten()

    def predict(self, instances):
        """
        @author: semercim
        :param instances:
        :return:
        """
        x_tensors = []
        for index in range(instances.shape[0]):
            x_tensors.append(sktensor.dtensor(instances[index, :].reshape(self.dimensions)))
        projected_x = self.get_projections(x_tensors)
        return self.predict_(projected_x)

    def predict_tensor(self, tensor_list):
        """
        @author: semercim
        :param tensor_list:
        :return:
        """
        projected_x = self.get_projections(tensor_list)
        return self.predict_(projected_x)

    def get_projections(self, tensors_list):
        """
        @author: semercim
        :param tensors_list:
        :return:
        """
        projected = copy.deepcopy(tensors_list)
        num_instances = len(tensors_list)
        for mode in range(self.num_ways):
            for index in range(num_instances):
                projected[index] = projected[index].ttm(self.projections[mode].T, mode)

        projections = []
        for index in range(num_instances):
            projections.append(projected[index].flatten())
        projections = np.array(projections)

        return projections

    def get_projected_tensors(self, tensors_list):
        """
        @author: semercim
        :param tensors_list:
        :return:
        """
        projected = copy.deepcopy(tensors_list)
        num_instances = len(tensors_list)
        for mode in range(self.num_ways):
            for index in range(num_instances):
                projected[index] = projected[index].ttm(self.projections[mode].T, mode)

        return projected

    def reconstruct_org(self, projected_tensors_list):
        """
        @author: semercim
        :param projected_tensors_list:
        :return:
        """
        reconstructed = copy.deepcopy(projected_tensors_list)
        num_instances = len(projected_tensors_list)
        for mode in range(self.num_ways):
            for index in range(num_instances):
                reconstructed[index] = reconstructed[index].ttm(self.projections[mode],
                                                                mode)

        return reconstructed

    def reconstruct(self, projected_tensors_list):
        """
        @author: semercim
        :param projected_tensors_list:
        :return:
        """
        return self.reconstruct_pinv(projected_tensors_list)

    def reconstruct_pinv(self, projected_tensors_list):
        """
        @author: semercim
        :param projected_tensors_list:
        :return:
        """
        reconstructed = copy.deepcopy(projected_tensors_list)
        num_instances = len(projected_tensors_list)
        for mode in range(self.num_ways):
            inv_projections = np.linalg.pinv(self.projections[mode].T)
            for index in range(num_instances):
                reconstructed[index] = reconstructed[index].ttm(inv_projections, mode)

        return reconstructed

    def set_result_set(self, result_set_):
        """
        @author: semercim
        :param result_set_:
        :return:
        """

        self.result_set = 'min_loss'
        if result_set_ not in ['min_loss', 'min_pair', 'min_impostors']:
            print("The initialization is not one of 'min_loss', 'min_pair', 'min_impostors' ")
            print("Setting min_loss as the default result_set")
        else:
            self.result_set = result_set_

        if self.result_set == 'min_loss':
            self.projections = copy.deepcopy(self.min_loss_projections)

        if self.result_set == 'min_pair':
            self.projections = copy.deepcopy(self.min_pair_projections)

        if self.result_set == 'min_impostors':
            self.projections = copy.deepcopy(self.min_impostors_projections)

    def set_max_iter(self, max_iter_):
        """
        @author: semercim
        :param max_iter_:
        :return:
        """
        self.max_iter = max_iter_

    def set_dimensions(self, dimensions_):
        """
        @author: semercim
        :param dimensions_:
        :return:
        """
        self.dimensions = dimensions_
        self.num_ways = len(dimensions_)

    def set_num_neighbors(self, num_neighbors_):
        """
        @author: semercim
        :param num_neighbors_:
        :return:
        """
        self.num_neighbors = num_neighbors_

    def get_projection_matrices(self):
        """
        @author: semercim
        :return:
        """
        return copy.deepcopy(self.projections)

    def set_coefficients(self, mu_=1.0, beta_=1.0, gamma_=1.0, alpha_=1.0, margin_=1.0):
        """
        @author: semercim
        :param mu_:
        :param beta_:
        :param gamma_:
        :param alpha_:
        :param margin_:
        :return:
        """
        self.mu = mu_
        self.beta = beta_
        self.gamma = gamma_
        self.alpha = alpha_
        self.margin = margin_


if __name__ == "__main__":
    print("This file is not supposed to be run as a standalone file.")
