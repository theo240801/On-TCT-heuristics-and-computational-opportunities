import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def scaffold_update(grads_data, targets, theta_client, h_i_client_pre, theta_global,
                    M=200, lr_local=0.00001):
    # set up data / eNTK
    grads_data = grads_data.float() #supprimer .cuda()
    targets = targets #supprimer .cuda()

    # compute transformed onehot label
    targets_onehot = F.one_hot(targets, num_classes=10) - (1.0 / 10.0) #supprimer .cuda()
    num_samples = targets_onehot.shape[0]

    # compute updates
    h_i_client_update = h_i_client_pre + (1 / (M * lr_local)) * (theta_global - theta_client)
    theta_hat_local = (theta_global) * 1.0

    # local gd
    print('local gd ')
    for local_iter in range(M):
        theta_hat_local -= lr_local * ((1.0 / num_samples) * grads_data.t() @ (grads_data @ theta_hat_local - targets_onehot) - h_i_client_update)
        print('num local iter :', local_iter, end='\r')


    del targets
    del grads_data
    torch.cuda.empty_cache()
    return theta_hat_local, h_i_client_update