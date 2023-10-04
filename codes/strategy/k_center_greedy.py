import torch
from get_ddis import *
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances

class kCenterGreedy:
  def __init__(self, X, already_selected, metric='euclidean'):
    self.X = X
    #self.y = y
    self.metric = metric 
    self.n_obs = self.X.shape[0] 
    self.already_selected = already_selected
    self.min_distances = None

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    if reset_dist:
      self.min_distances = None
    
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
      
    if cluster_centers:
      dist = pairwise_distances(self.X.cpu(), self.X[cluster_centers].cpu(), metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)
    
  def select_batch_(self, N):
    #try:
      #self.X = model.transform(self.X)
    
    self.update_distances(self.already_selected, only_new=False, reset_dist=True)
    
    #except:
    #  self.update_distances(self.already_selected, only_new=True, reset_dist=False)

    new_batch = []
    for _ in range(N):
    
      if not self.already_selected: 
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      assert ind not in self.already_selected
    
      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
      self.already_selected.append(ind)

    return new_batch

def k_center_greedy(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels):
    
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

        queried_list = queried.tolist()
        selector = kCenterGreedy(feature_training_set, queried_list)

        if idx_it != num_iters - 1:
            sampling_num = query_batch_size
        elif idx_it == num_iters - 1:
            sampling_num = len(rels_train) - len(queried)
        
        if idx_it == num_iters - 1:
            queried = torch.arange(len(h_train))
            print("last iter get a new query batch!")
            num_queries.append(len(h_train))
        else:
            new_queries = selector.select_batch_(sampling_num)
            queried = torch.tensor(selector.already_selected)
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
