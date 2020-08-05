# Large Margin Tensor Decomposition - LMTD

The code provided here is distributed as-is and is run with python 3.7.

FLMTD_Basic_Runner.py contain an example to run the proposed decomposition: LMTD-F. A similar runner file could also be written for LMTD-C. For the parameters, please refer to the journal paper. Note that, the case of 5 instances for each class during training means that each instance has 4 neighbors (--num_neighbors 4) 

FLMTD_Basic_Runner.py --mu 1.0 --beta 1.0 --gamma 1.0 --alpha 1.0 --max_iter 500 --num_components 7 7 7 --margin 32 --learning_rate 0.01 --num_neighbors 4 --verbose True --initialization mda --result_set min_loss --init_reduced_dim True --reconstruction True --hard_orthogonality True --starting_fold 1 --stopping_fold 21 --file_header ETH80 --data_path Datasets/ETH80/ETH80_normalized_32x32.mat --fold_indices_path Datasets/ETH80/5Train/


If you use the algorithm implemented in this repository, please cite the following paper:

Murat Semerci, Ali Taylan Cemgil, Bulent Sankur, "Discriminative tensor decomposition with large margin", Digital Signal Processing, Volume 95, 2019, 102584, ISSN 1051-2004, https://doi.org/10.1016/j.dsp.2019.102584. (http://www.sciencedirect.com/science/article/pii/S1051200419301307)