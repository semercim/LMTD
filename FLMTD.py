# -*- coding: utf-8 -*-
"""
@author: semercim (murat.semerci@boun.edu.tr)
"""

import copy
import numpy as np
from scipy import linalg as spla
from LMTD import LMTD


class FLMTD(LMTD):
    """
    This class implements FLMTD
    """
    def __init__(self, num_components=None, max_iter=100, dimensions=None,
                 mu=1.0, beta=1.0, gamma=1.0, alpha=None, margin=None,
                 learning_rate=0.01, num_neighbors=1, update_iter=1, verbose=False,
                 initialization='randomized', result_set='min_loss',
                 init_reduced_dim=True, reconstruction=True, hard_orthogonality=True):
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
        :param reconstruction: done or not,  default True
        :param hard_orthogonality: soft or hard orthogonality constraint, default True
        """
        super(FLMTD, self).__init__(num_components, max_iter, dimensions, mu, beta, gamma, alpha,
                                    margin, learning_rate, num_neighbors, update_iter, verbose,
                                    initialization, result_set, init_reduced_dim)
        self._name = 'FLMTD'
        self.hard_orthogonality = hard_orthogonality
        self.reconstruction = reconstruction

    def calculate_loss(self, tensors_list, core_tensors, target_neighbors):
        """
        @author: semercim
        :param tensors_list: list of input tensors (N instances)
        :param core_tensors: the core tensors (N instances)
        :param target_neighbors: the target neighbors (N x k)
        :return:
            loss: the value of loss function
        """
        num_instances = len(tensors_list)
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0
        loss4 = 0.0
        num_impostor_tuples = 0
        hat_tensors_list = copy.deepcopy(core_tensors)
        # reconstruct input tensors from the core tensors
        for index in range(num_instances):
            for mode in range(self.num_ways):
                hat_tensors_list[index] = hat_tensors_list[index].ttm(self.projections[mode], mode)

        # vectorize them for easening the distance calculations
        vectorized_tensors = []
        for tensor in hat_tensors_list:
            vectorized_tensors.append(copy.deepcopy(tensor).flatten())
        vectorized_tensors = np.array(vectorized_tensors)

        for index in range(num_instances):
            # calculate the reconstruction error is demanded
            if self.reconstruction or (self.mu != 0.0):
                loss1 += np.sum((tensors_list[index] - hat_tensors_list[index]) ** 2.0)
            # find the impostors in the current iteration
            violated, tar_dists, distances = self.find_impostors(vectorized_tensors,
                                                                 index,
                                                                 target_neighbors)
            # calculate the target distances
            for target in target_neighbors[index, :]:
                loss2 += np.sum((hat_tensors_list[index] - hat_tensors_list[target]) ** 2.0)
                if len(violated[0]):
                    target_distance = tar_dists[0,
                                                np.where(target == target_neighbors[index, :])
                                                [0]]
                    impostors = np.maximum(target_distance - distances[violated[0]], 0.0)
                    # calculate the impostor loss
                    loss3 += np.sum(impostors)
                    num_impostor_tuples += np.sum(impostors > 0.0)
        # calculate the soft orthogonality loss
        if not self.hard_orthogonality:
            for mode in range(self.num_ways):
                T = (np.dot(self.projections[mode][:, :self.num_components[mode]],
                            self.projections[mode][:, :self.num_components[mode]].T) -
                     np.eye(self.projections[mode][:, :self.num_components[mode]].shape[0]))
                loss4 += np.sum(T ** 2.0)

        loss = self.mu * loss1 + self.beta * loss2 + self.gamma * loss3 + self.alpha * loss4

        # if self.verbose:
        #     print("First Loss Term (loss1): " + str(loss1))
        #     print("Second Loss Term (loss2): " + str(loss2))
        #     print("Third Loss Term (loss3): " + str(loss3))
        #     print("Fourth Loss Term (loss4): " + str(loss4))
        #     print("Total Loss: " + str(loss))

        print("Total Impostor Tuples: " + str(num_impostor_tuples))
        return loss, num_impostor_tuples

    def update_projections(self, tensors_list, target_neighbors):
        """
        @author: semercim
        :param tensors_list: list of input tensors (N instances)
        :param target_neighbors: the target neighbors indices (N x k)
        :return:
            us: the list of projection matrices
        """
        # Get the number of instances and dimension

        num_instances = len(tensors_list)

        for _ in range(self.update_iter):
            counts = np.zeros((num_instances, num_instances))
            for mode in range(self.num_ways):
                Ts, Ss = self.get_intermediate_variable()
                T = Ts[mode]
                S = Ss[mode]
                Ws = []
                hat_tensors_list = []

                for index in range(num_instances):
                    # calculate the W matrices for each instance
                    Ws.append(np.dot(self.unfolded[mode][index], S))
                    # calculate the reconstructed matrix for each instance
                    hat_tensors_list.append(tensors_list[index])
                    for dimension in range(self.num_ways):
                        hat_tensors_list[index] = hat_tensors_list[index].ttm(
                            Ts[dimension], dimension)
                    # vectorize each tensor for the distance calculations
                vectorized_tensors = []
                for tensor in hat_tensors_list:
                    vectorized_tensors.append(copy.deepcopy(tensor).flatten('F'))
                vectorized_tensors = np.array(vectorized_tensors)

                loss1 = 0.0
                loss23 = 0.0
                for index in range(num_instances):
                    # Calculate first loss term gradient. The reconstruction error.
                    if self.reconstruction or (self.mu != 0.0):
                        Vii = np.dot(Ws[index], Ws[index].T)
                        WiXi = np.dot(Ws[index], self.unfolded[mode][index].T)
                        ViiT = np.dot(Vii, T)
                        loss1 += (- WiXi - WiXi.T + ViiT + ViiT.T)
                    violated, tar_dists, distances = self.find_impostors(vectorized_tensors,
                                                                         index, target_neighbors)
                    for target in target_neighbors[index, :]:
                        # Calculate the second loss term gradient.
                        # The distance to the target neighbor.
                        counts[index, index] += self.beta
                        counts[index, target] -= self.beta
                        counts[target, index] -= self.beta
                        counts[target, target] += self.beta
                        # Calculate the third loss term gradient.
                        # Find and drive away the impostors.
                        if len(violated[0]):
                            target_distance = tar_dists[0,
                                                        np.where(target ==
                                                                 target_neighbors[index, :])
                                                        [0]]
                            for impostor in violated[0]:
                                impostor_distance = distances[impostor]
                                if (self.y_[impostor] == self.y_[index] or
                                        impostor_distance > target_distance):
                                    continue

                                counts[index, target] -= self.gamma
                                counts[target, index] -= self.gamma
                                counts[target, target] += self.gamma

                                counts[index, impostor] += self.gamma
                                counts[impostor, index] += self.gamma
                                counts[impostor, impostor] -= self.gamma

                for index in range(num_instances):
                    if counts[index, index] != 0:
                        loss23 += counts[index, index] * np.dot(Ws[index], Ws[index].T)
                    for index2 in range(index+1, num_instances):
                        if counts[index, index2] != 0:
                            inner_matrix_ = np.dot(Ws[index], Ws[index2].T)
                            loss23 += counts[index, index2] * (inner_matrix_ + inner_matrix_.T)
                loss23 = np.dot(loss23, T)
                loss23 += loss23.T
                loss23 = 2 * np.dot(loss23, self.projections[mode])
                if self.reconstruction or (self.mu != 0.0):
                    loss1 = 2 * np.dot(loss1, self.projections[mode])
                # Calculate the fourth loss term gradient. How far way the projection matrices
                # from the identity constraint.
                loss4 = 0.0
                if not self.hard_orthogonality:
                    loss4 = 4 * np.dot(T - np.eye(T.shape[0]), self.projections[mode])

                grad = self.mu * loss1 + loss23 + self.alpha * loss4
                grad = grad / np.max(np.abs(np.array(grad)))
                self.projections[mode] -= self.learning_rate * grad

                if self.hard_orthogonality:
                    # self.projections[mode] /= np.linalg.norm(self.projections[mode], axis=0)
                    self.projections[mode], _ = np.linalg.qr(self.projections[mode])

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

        super(FLMTD, self).fit_tensor(tensors_list, labels, max_iter)
        for mode in range(self.num_ways):
            self.projections[mode] = self.projections[mode][:, :self.num_components[mode]]


if __name__ == "__main__":
    print("This file is not supposed to be run as a standalone file.")
