# -*- coding: utf-8 -*-
"""
@author: semercim (murat.semerci@boun.edu.tr)
"""

import scipy.io as sio
import numpy as np
import sktensor
from FLMTD import FLMTD
import argparse
from common import knn_classify_many, knn_classify0, sort_neighbors, average_precision_at_k


parent_parser = argparse.ArgumentParser()
parent_parser.add_argument('--dimensions', nargs='*', default=None)
parent_parser.add_argument('--mu', type=float, default=1.0)
parent_parser.add_argument('--beta', type=float, default=1.0)
parent_parser.add_argument('--gamma', type=float, default=1.0)
parent_parser.add_argument('--alpha', type=float, default=None)
parent_parser.add_argument('--max_iter', type=int, default=500)
parent_parser.add_argument('--num_components', nargs='*', default=[8, 6, 8])
parent_parser.add_argument('--margin', type=float, default=None)
parent_parser.add_argument('--learning_rate', type=float, default=0.01)
parent_parser.add_argument('--num_neighbors', type=int, default=1)
parent_parser.add_argument('--verbose', type=bool, default=False)
parent_parser.add_argument('--initialization', type=str, default='randomized')
parent_parser.add_argument('--result_set', type=str, default='min_loss')
parent_parser.add_argument('--init_reduced_dim', type=bool, default=True)
parent_parser.add_argument('--reconstruction', type=bool, default=True)
parent_parser.add_argument('--hard_orthogonality', type=bool, default=True)
parent_parser.add_argument('--starting_fold', type=int, default=1)
parent_parser.add_argument('--stopping_fold', type=int, default=11)
parent_parser.add_argument('--file_header', type=str, default='./')
parent_parser.add_argument('--data_path', type=str, default='./')
parent_parser.add_argument('--fold_indices_path', type=str, default='./')


def main():
    """
    @author: semercim
    A common runner function for running the method.
    :return:
    """

    args = parent_parser.parse_args()
    print(args)

    if args.num_neighbors == 0:
        knn_classify = knn_classify0
    else:
        knn_classify = knn_classify_many

    use_flmtd = True
    use_flmtd_reconstruct = True

    file_header = args.file_header

    test_acc_results = []
    test_mAP_results = []

    test_reconstructed_acc_results = []
    test_reconstructed_mAP_results = []

    parameters = {'num_components': map(int, args.num_components), 'max_iter': args.max_iter,
                  'dimensions': args.dimensions, 'mu': args.mu, 'beta': args.beta,
                  'gamma': args.gamma, 'alpha': args.alpha,
                  'margin': args.margin, 'learning_rate': args.learning_rate,
                  'num_neighbors': args.num_neighbors, 'verbose': args.verbose,
                  'initialization': args.initialization,
                  'result_set': args.result_set,
                  'init_reduced_dim': args.init_reduced_dim,
                  'reconstruction': args.reconstruction,
                  'hard_orthogonality': args.hard_orthogonality}

    decomposers = {'FLMTD': use_flmtd,
                   'FLMTD_Reconstructed': use_flmtd_reconstruct}

    # Load the dataset
    mat_contents = sio.loadmat(args.data_path)
    data = mat_contents['data']
    labels = mat_contents['labels'][0]
    num_instances = data.shape[0]
    tensor_list = [sktensor.dtensor(data[i]) for i in range(num_instances)]

    num_components = list(map(int, parameters['num_components']))
    for iteration in range(args.starting_fold, args.stopping_fold):
        print('Iteration: ', iteration)
        # Get the training and test data set indices for this round
        mat_contents = sio.loadmat(args.fold_indices_path + 'fold{0}.mat'.format(iteration))
        test_indices = mat_contents['testIdx'][0]
        train_indices = mat_contents['trainIdx'][0]
        labels_train = np.squeeze(labels[train_indices])
        labels_test = np.squeeze(labels[test_indices])

        # Get the training and test data sets as list
        tensor_list_train = []
        for index in train_indices:
            tensor_list_train.append(tensor_list[int(index)])
        tensor_list_test = []
        for index in test_indices:
            tensor_list_test.append(tensor_list[int(index)])

        ##########################################################################################
        # FLMTD is run in this section
        if decomposers['FLMTD']:
            # np.random.seed(0)
            flmtd_ = FLMTD(num_components=num_components,
                           max_iter=parameters['max_iter'],
                           dimensions=parameters['dimensions'], mu=parameters['mu'],
                           beta=parameters['beta'], gamma=parameters['gamma'],
                           alpha=parameters['alpha'], margin=parameters['margin'],
                           learning_rate=parameters['learning_rate'],
                           num_neighbors=parameters['num_neighbors'],
                           verbose=parameters['verbose'],
                           initialization=parameters['initialization'],
                           result_set=parameters['result_set'],
                           init_reduced_dim=parameters['init_reduced_dim'],
                           reconstruction=parameters['reconstruction'],
                           hard_orthogonality=parameters['hard_orthogonality'])
            flmtd_.fit_tensor(tensor_list_train, labels_train)

            # knn ve mAP results over core (projected) tensors
            ### You could change the result set here: min_loss (default), min_pair, min_impostors
            # flmtd_.set_result_set('min_loss')
            ####
            test_projected = flmtd_.get_projections(tensor_list_test)
            train_projected = flmtd_.get_projections(tensor_list_train)

            projected_training = []
            for tensor in train_projected:
                projected_training.append(tensor.flatten())

            projected_test = []
            for tensor in test_projected:
                projected_test.append(tensor.flatten())

            training_result, test_result = knn_classify(projected_training,
                                                        labels_train,
                                                        projected_test,
                                                        labels_test,
                                                        args.num_neighbors)

            sn, sd, indices = sort_neighbors(np.array(projected_training), labels_train, np.array(projected_test))
            matched, precisions, av_precisions = average_precision_at_k(sn, labels_test)
            mAP = np.mean(av_precisions)
            ####

            print('FLMTD Test mAP result: ', mAP)
            test_mAP_results.append(mAP)
            print('FLMTD Test Accuracy result: ', test_result)
            test_acc_results.append(test_result)

            if decomposers['FLMTD_Reconstructed']:
                # knn ve mAP results over reconstructed tensors
                ### You could change the result set here: min_loss (default), min_pair, min_impostors
                # flmtd_.set_result_set('min_loss')
                projected_tensors_train = flmtd_.get_projected_tensors(tensor_list_train)
                projected_tensors_test = flmtd_.get_projected_tensors(tensor_list_test)
                reconstructed_tensors_train = flmtd_.reconstruct(projected_tensors_train)
                reconstructed_tensors_test = flmtd_.reconstruct(projected_tensors_test)

                reconstructed_training = []
                for tensor in reconstructed_tensors_train:
                    reconstructed_training.append(tensor.flatten())

                reconstructed_test = []
                for tensor in reconstructed_tensors_test:
                    reconstructed_test.append(tensor.flatten())

                training_result_reconstructed, test_result_reconstructed = knn_classify(reconstructed_training,
                                                                                        labels_train,
                                                                                        reconstructed_test,
                                                                                        labels_test,
                                                                                        args.num_neighbors)
                sn, sd, indices = sort_neighbors(np.array(reconstructed_training), labels_train,
                                                 np.array(reconstructed_test))
                matched, precisions, av_precisions = average_precision_at_k(sn, labels_test)
                mAP = np.mean(av_precisions)
                print('FLMTD Test Reconstructed mAP result: ', mAP)
                test_reconstructed_mAP_results.append(mAP)
                print('FLMTD Test Reconstructed Test Accuracy result: ', test_result_reconstructed)
                test_reconstructed_acc_results.append(test_result)

    ##############################################################################
    # Print the results of FLMTD
    if decomposers['FLMTD']:
        overall_acc_results = np.array(test_acc_results)
        overall_mAP_results = np.array(test_mAP_results)
        print('FLMTD Overall Test Accuracy Mean: ', np.mean(overall_acc_results),
              ' Std: ', np.std(overall_acc_results))
        print('FLMTD Overall mAP Mean: ', np.mean(overall_mAP_results),
              ' Std: ', np.std(overall_mAP_results))

        if decomposers['FLMTD_Reconstructed']:
            overall_reconstructed_acc_results = np.array(test_reconstructed_acc_results)
            overall_reconstructed_mAP_results = np.array(test_reconstructed_mAP_results)
            print('FLMTD Reconstructed Overall Test Accuracy  Mean: ',
                  np.mean(overall_reconstructed_acc_results),
                  ' Std: ', np.std(overall_reconstructed_acc_results))
            print('FLMTD Reconstructed Overall mAP  Mean: ',
                  np.mean(overall_reconstructed_mAP_results),
                  ' Std: ', np.std(overall_reconstructed_mAP_results))


if __name__ == "__main__":
    main()
