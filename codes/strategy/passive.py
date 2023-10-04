import torch
from get_ddis import *
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os

import wandb
import logging

def passive(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels):
     
    if encoder == 'DSN-DDI':
        h_train, t_train, b_train, rels_train, labels_train,\
            h_test, t_test, b_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
    else:
        h_train, t_train, rels_train, labels_train,\
            h_test, t_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
    
    seed = 321
    rnd = torch.Generator()
    rnd.manual_seed(seed)
    random_idx = torch.randperm(labels_train.size(0), generator=rnd)
    
    
    for iter in range(num_iters):
        if iter == 0:
            idx_this_iter = random_idx[0:min(init_batch_size, len(rels_train))]
        else:
            idx_this_iter = random_idx[0:min(iter * query_batch_size + init_batch_size, len(rels_train))]
        h_train_iter = [h_train[idx] for idx in idx_this_iter]
        t_train_iter = [t_train[idx] for idx in idx_this_iter]
        rels_train_iter = [rels_train[idx] for idx in idx_this_iter]
        labels_train_iter = [labels_train[idx] for idx in idx_this_iter]
                
        ## train
        if encoder == 'DSN-DDI': 
            b_train_iter = [b_train[idx] for idx in idx_this_iter]
            trainer_train(encoder, trainer, h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, iter, b_train_iter)
        else:     
            trainer_train(encoder, trainer, h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, iter)
        
        ## sample pool
        with torch.no_grad():
            if encoder == 'DSN-DDI':
                scores = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train)
            else:
                scores = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train)
             
            model_acc, weighted_precision, weighted_recall, weighted_f1,\
                    hit3, hit5, hit10,\
                    all_class_acc, kappa = compute_all_metrics(rels_train, scores)
                    
        ## testing set
        with torch.no_grad():
            if encoder == 'DSN-DDI':
                scores_test = trainer_test(encoder, trainer, h_test, t_test, rels_test, labels_test, b_test)
            else:
                scores_test = trainer_test(encoder, trainer, h_test, t_test, rels_test, labels_test)
            
            model_acc_test, weighted_precision_test, weighted_recall_test, weighted_f1_test,\
                    hit3_test, hit5_test, hit10_test,\
                    all_class_acc_test, kappa_test = compute_all_metrics(rels_test, scores_test)
            
        print(f'Iter:{iter}, Number of labels:{len(h_train_iter)}, Testing acc:{model_acc_test}\n')
        
    return trainer.model