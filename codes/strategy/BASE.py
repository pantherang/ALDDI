import torch
from get_ddis import *
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def BASE(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels):
    
    if encoder == 'DSN-DDI':
        h_train, t_train, b_train, rels_train, labels_train,\
            h_test, t_test, b_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
    else:
        h_train, t_train, rels_train, labels_train,\
            h_test, t_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
    
    seed = 321
    rnd = torch.Generator()
    rnd.manual_seed(seed)
        
    ########## iter 0
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
            scores, f_tensor, min_margins, per_class_margins = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train, ret_features = True, ret_margins = True)
        else:
            scores, f_tensor, min_margins, per_class_margins = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, ret_features = True, ret_margins = True)
        
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
        new_queries = []
        queried_list = list(queried)

        if idx_it != num_iters - 1:
            sampling_num = query_batch_size
        elif idx_it == num_iters - 1:
            sampling_num = len(rels_train) - len(queried)
    
        for c in tqdm(range(num_rels)):
            cur_class_query_count = int(sampling_num / num_rels) + int(c < sampling_num % num_rels)
            if cur_class_query_count == 0:
                continue
            cur_class_distance = torch.where(torch.argmax(scores, dim=-1) == c, 
                                             min_margins,
                                             per_class_margins[:, c].squeeze())

            cur_labeled_idxs = list(torch.argsort(cur_class_distance, descending=False).cpu().numpy().astype(int))
            count = 0
            for idx in cur_labeled_idxs:
                if idx not in queried_list:
                    queried_list.append(idx)
                    new_queries.append(idx)
                    #queried_set.add(idx)
                    count += 1
                if count == cur_class_query_count:
                    break

        print("get a new query batch!")

        num_queries.append(num_queries[-1] + len(new_queries))
        queried = torch.tensor(queried_list)
        
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
                scores, f_tensor, min_margins, per_class_margins = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train, ret_features = True, ret_margins = True)
            else:
                scores, f_tensor, min_margins, per_class_margins = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, ret_features = True, ret_margins = True)
            
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
