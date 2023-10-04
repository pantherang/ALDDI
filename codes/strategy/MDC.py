import torch
from get_ddis import *
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans

def my_KMeans(emb, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(emb)
    '''
    clustered_samples = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clustered_samples[label].append(i)
    print(clustered_samples)
    return clustered_samples
    '''
    return labels

def MDC(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels):
    
    if encoder == 'DSN-DDI':
        h_train, t_train, b_train, rels_train, labels_train,\
            h_test, t_test, b_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
    else:
        h_train, t_train, rels_train, labels_train,\
            h_test, t_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
        
    seed = 321
    rnd = torch.Generator()
    rnd.manual_seed(seed)
        
    ######### iter 0
    idx_it = 0
    init_batch = torch.randperm(labels_train.size(0), generator=rnd)[:init_batch_size]
    h_train_iter = [h_train[idx] for idx in init_batch]
    t_train_iter = [t_train[idx] for idx in init_batch]
    rels_train_iter = [rels_train[idx] for idx in init_batch]
    labels_train_iter = [labels_train[idx] for idx in init_batch]
    
    # iter 0 train
    if encoder == 'DSN-DDI': 
        b_train_iter = [b_train[idx] for idx in init_batch]
        trainer_train(encoder, trainer, h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, idx_it, b_train_iter)
    else:     
        trainer_train(encoder, trainer, h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, idx_it)
    # iter 0 inference
    with torch.no_grad():
        if encoder == 'DSN-DDI':
            scores, feature_training_set = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train, ret_features = True)
        else:
            scores, feature_training_set = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, ret_features = True)
        
        model_acc, weighted_precision, weighted_recall, weighted_f1,\
                    hit3, hit5, hit10,\
                    all_class_acc, kappa = compute_all_metrics(rels_train, scores)

    # iter 0 test
    with torch.no_grad():
        if encoder == 'DSN-DDI':
            scores_test = trainer_test(encoder, trainer, h_test, t_test, rels_test, labels_test, b_test)
        else:
            scores_test = trainer_test(encoder, trainer, h_test, t_test, rels_test, labels_test)
        
        model_acc_test, weighted_precision_test, weighted_recall_test, weighted_f1_test,\
                    hit3_test, hit5_test, hit10_test,\
                    all_class_acc_test, kappa_test = compute_all_metrics(rels_test, scores_test)
                    
    print(f'Iter:{0}, Number of labels:{len(h_train_iter)}, Testing acc:{model_acc_test}\n')
    
    queried = init_batch 
    num_queries = [init_batch_size] 

    while num_queries[-1] < len(rels_train):
        idx_it += 1

        if idx_it != num_iters - 1:
            sampling_num = query_batch_size
        elif idx_it == num_iters - 1:
            sampling_num = len(rels_train) - len(queried)
        
        probs = F.softmax(scores, dim=1)
        sorted_pred = torch.sort(probs, dim=-1, descending=True)[0]
        margins = sorted_pred[:, 0] - sorted_pred[:, 1]
        uncertain_idxs = np.argsort(margins.cpu().numpy())        
        cluster_method = "KMeans"
        feature_training_set = feature_training_set.detach().cpu()
        n_clusters = 86 
        if cluster_method == "KMeans":
            cluster_idxs = my_KMeans(feature_training_set, n_clusters)
        print(f"use the cluster method:{cluster_method}")
        
        clusters = [[] for _ in range(np.max(cluster_idxs) + 1)]
        num_batch_points = 0
        cluster_batch_size = int(2 * sampling_num)
        for idx in uncertain_idxs:
            if idx not in queried:
                clusters[cluster_idxs[idx]].append(idx)
                num_batch_points += 1
            if num_batch_points == cluster_batch_size:
                break
            
        cluster_unsorted_idxs = np.arange(len(clusters))
        c_idx = 0 
        new_queries = []
        queried_list = list(queried)
        while len(new_queries) < sampling_num:
            cluster_idx = cluster_unsorted_idxs[c_idx]
            if len(clusters[cluster_idx]) != 0:                
                idx = clusters[cluster_idx].pop(0) 
                queried_list.append(idx)
                new_queries.append(idx)
            c_idx = (c_idx + 1) % len(clusters) 
        
        queried = torch.tensor(queried_list)
        print("get a new query batch!")
        num_queries.append(num_queries[-1] + len(new_queries))
        
        # update
        h_train_iter = [h_train[idx] for idx in queried]
        t_train_iter = [t_train[idx] for idx in queried]
        rels_train_iter = [rels_train[idx] for idx in queried]
        labels_train_iter = [labels_train[idx] for idx in queried]
        
        if encoder == 'DSN-DDI': 
            b_train_iter = [b_train[idx] for idx in queried]
            trainer_train(encoder, trainer, h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, idx_it, b_train_iter)
        else:     
            trainer_train(encoder, trainer, h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, idx_it)
        
        with torch.no_grad():
            if encoder == 'DSN-DDI':
                scores, feature_training_set = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train, ret_features = True)
            else:
                scores, feature_training_set = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, ret_features = True)
               
            model_acc, weighted_precision, weighted_recall, weighted_f1,\
                    hit3, hit5, hit10,\
                    all_class_acc, kappa = compute_all_metrics(rels_train, scores)
                
        with torch.no_grad():
            if encoder == 'DSN-DDI':
                scores_test = trainer_test(encoder, trainer, h_test, t_test, rels_test, labels_test, b_test)
            else:
                scores_test = trainer_test(encoder, trainer, h_test, t_test, rels_test, labels_test)
            
            model_acc_test, weighted_precision_test, weighted_recall_test, weighted_f1_test,\
                    hit3_test, hit5_test, hit10_test,\
                    all_class_acc_test, kappa_test = compute_all_metrics(rels_test, scores_test)
                    
        print(f'Iter:{idx_it}, Number of labels:{len(h_train_iter)}, Testing acc:{model_acc_test}\n')
        
    return trainer.model