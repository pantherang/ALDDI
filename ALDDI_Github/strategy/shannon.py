import torch
from get_ddis import *
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def shannon(tri_train, tri_test, encoder, dataset, trainer, query_batch_size, num_iters, num_rels):
    
    if encoder == 'DSN-DDI':
        h_train, t_train, b_train, rels_train, labels_train,\
            h_test, t_test, b_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
    else:
        h_train, t_train, rels_train, labels_train,\
            h_test, t_test, rels_test, labels_test = get_whole_data(encoder, tri_train, tri_test)
        
    seed = 321
    rnd = torch.Generator()
    rnd.manual_seed(seed)
        
    idx_it = 0
    init_batch = torch.randperm(labels_train.size(0), generator=rnd)[:query_batch_size]
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
            scores = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train)
        else:
            scores = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train)
        
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
    num_queries = [query_batch_size] 
    
    while num_queries[-1] < len(rels_train):
        idx_it += 1
        new_queries = []
        if idx_it != num_iters - 1:
            sampling_num = query_batch_size
        elif idx_it == num_iters - 1:
            sampling_num = len(rels_train) - len(queried)
            

        probs = F.softmax(scores, dim=1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
        sorted_indices = torch.argsort(-entropy)
        indices_to_keep = ~torch.isin(sorted_indices.detach().cpu(), queried.detach().cpu()).cpu()
        filtered_sorted_indices = sorted_indices[indices_to_keep].tolist()
        new_queries = filtered_sorted_indices[:sampling_num]         
        print("get a new query batch!")
        num_queries.append(num_queries[-1] + len(new_queries)) 
        new_queries = torch.from_numpy(np.array(new_queries))
        queried = torch.cat([queried, new_queries], dim=0) 
        
        ## updata training data
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
                scores = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train)
            else:
                scores = trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train)
               
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
