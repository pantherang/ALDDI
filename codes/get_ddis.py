import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_all_ddi_triplets_drugbank(df):
    data_df = df
    head_list = []
    tail_list = []
    label_list = []
    rel_list = []
    for index, row in data_df.iterrows():
        Drug1_ID, Drug2_ID, Y = row['Drug1_ID'], row['Drug2_ID'], row['Y']
        head_list.append(Drug1_ID)
        tail_list.append(Drug2_ID)
        rel_list.append(torch.LongTensor([Y]))
        label_list.append(torch.FloatTensor([1]))
    rel = torch.cat(rel_list, dim=0)
    label = torch.cat(label_list, dim=0)
    print("Number of all the drugbank ddi triplets:{}".format(len(rel)))
    return head_list, tail_list, rel, label

def get_triplet_list(rels, num_rels):
    triplet_list = [[] for _ in range(num_rels)] 
    for i in range(len(rels)):
        triplet_list[rels[i]].append(i)
    return triplet_list

# metrics
def compute_all_metrics(true_labels, scores):

    pred_labels = scores.argmax(dim=1)

    true_labels_np = true_labels.cpu().numpy()
    pred_labels_np = pred_labels.cpu().numpy()
    # acc
    accuracy = metrics.accuracy_score(true_labels_np, pred_labels_np)
    # weighted
    weighted_precision = metrics.precision_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
    weighted_recall = metrics.recall_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
    weighted_f1 = metrics.f1_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)   
    # hit
    hit3, hit5, hit10 = cal_hit(true_labels, scores)
    # class acc
    all_class_acc = class_acc(true_labels_np, pred_labels_np)
    # kappa
    kappa = cohen_kappa_score(true_labels_np, pred_labels_np)
    return accuracy, weighted_precision, weighted_recall, weighted_f1,\
                    hit3, hit5, hit10,\
                    all_class_acc, kappa

# Hit
def cal_hit(true_labels, scores):
    softmax_scores = F.softmax(scores, dim=1)
    topk_indices = torch.topk(softmax_scores, k=10).indices
    #print(topk_indices)
    #for i in range(5):
    #    sorted_indices = torch.argsort(softmax_scores[i], descending=True)
    #    print(sorted_indices)
    true_labels_expanded = true_labels.view(-1, 1).expand_as(topk_indices)
    #print(true_labels_expanded)

    hit_matrix = topk_indices.to(scores.device) == true_labels_expanded.to(scores.device)
    #print(hit_matrix)
    
    hits_3 = hit_matrix[:,:3]
    hits_5 = hit_matrix[:,:5]
    hits_10 = hit_matrix[:,:10]
    
    hit3_accuracy = hits_3.sum().item() / hit_matrix.shape[0]
    hit5_accuracy = hits_5.sum().item() / hit_matrix.shape[0]
    hit10_accuracy =  hits_10.sum().item() / hit_matrix.shape[0]
    
    return hit3_accuracy, hit5_accuracy, hit10_accuracy

# class acc
def class_acc(truth_labels, pred_labels): 
    unique_labels, counts = np.unique(truth_labels, return_counts=True)
    label_counts = {}
    for label, label_count in zip(unique_labels, counts):
        label_counts[label] = label_count
    #print(label_counts)
    label_indices = {label: np.where(truth_labels == label)[0] for label in unique_labels}
    #print(label_indices)
    accuracies = {}
    for label in unique_labels:
        correct_count = np.sum(pred_labels[label_indices[label]] == label)
        total_count = label_counts[label]
        if total_count > 0:
            accuracies[label] = correct_count / total_count
        else:
            accuracies[label] = 0.0
    return accuracies
    
## some functions in different strategies
# get the whole training pool data before the first iter
def get_whole_data(encoder, tri_train, tri_test):
    if encoder == 'DSN-DDI': 
        h_train = tri_train[0]
        t_train = tri_train[1]
        b_train = tri_train[3]
        rels_train = tri_train[2]
        labels_train = tri_train[4]
        
        h_test = tri_test[0]
        t_test = tri_test[1]
        b_test = tri_test[3]
        rels_test = tri_test[2]
        labels_test = tri_test[4]
        
        return h_train, t_train, b_train, rels_train, labels_train, h_test, t_test, b_test, rels_test, labels_test
    else:
        h_train = tri_train[0]
        t_train = tri_train[1]
        rels_train = tri_train[2]
        labels_train = tri_train[3]
        
        h_test = tri_test[0]
        t_test = tri_test[1]
        rels_test = tri_test[2]
        labels_test = tri_test[3] 
        
    return h_train, t_train, rels_train, labels_train, h_test, t_test, rels_test, labels_test

# trainer.train
def trainer_train(encoder, trainer, h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, idx_it, b_train_iter = None):
    if encoder == 'DSN-DDI':
        trainer.train(h_train_iter, t_train_iter, rels_train_iter, b_train_iter, labels_train_iter, idx_it)
    else:
        trainer.train(h_train_iter, t_train_iter, rels_train_iter, labels_train_iter, idx_it)

# trainer.pred
def trainer_pred(encoder, trainer, h_train, t_train, rels_train, labels_train, b_train = None, ret_features = False, ret_margins = False):
    if ret_features and ret_margins: 
        if encoder == 'DSN-DDI':
            scores, f_tensor, min_margins, per_class_margins = trainer.pred(h_train, t_train, rels_train, b_train, labels_train, ret_features = True, ret_margins=True)
        else:
            scores, f_tensor, min_margins, per_class_margins = trainer.pred(h_train, t_train, rels_train, labels_train, ret_features = True, ret_margins=True)
        return scores, f_tensor, min_margins, per_class_margins
    
    elif ret_features: 
        if encoder == 'DSN-DDI':
            scores, features = trainer.pred(h_train, t_train, rels_train, b_train, labels_train, ret_features = True)
        else:
            scores, features = trainer.pred(h_train, t_train, rels_train, labels_train, ret_features = True)
        return scores, features
    
    else: 
        if encoder == 'DSN-DDI':
            scores = trainer.pred(h_train, t_train, rels_train, b_train, labels_train)
        else:
            scores = trainer.pred(h_train, t_train, rels_train, labels_train)
        return scores
    
# trainer.test
def trainer_test(encoder, trainer, h_test, t_test, rels_test, labels_test, b_test = None):
    if encoder == 'DSN-DDI':
        scores_test = trainer.test(h_test, t_test, rels_test, b_test, labels_test)
    else:
        scores_test = trainer.test(h_test, t_test, rels_test, labels_test)
    return scores_test
    
    
