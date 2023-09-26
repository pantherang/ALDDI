import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import tqdm
from DGNNDDI.models import *

from torch_geometric.data import Batch


class Passive_DGNNDDI_Trainer:
    def __init__(self, device, train_batch_size, train_num_epoch, num_rels):
        self.train_batch_size = train_batch_size
        self.device = device
        self.train_num_epoch = train_num_epoch
        self.num_rels = num_rels

    def train(self, h_train, t_train, rels_train, labels_train, cur_idx):
        # model.py
        self.model = train_DGNN(h_train, t_train, rels_train, labels_train, num_rels=self.num_rels, train_batch_size=self.train_batch_size, train_num_epoch = self.train_num_epoch, device = self.device)
        
    def pred(self, h_train, t_train, rels_train, labels_train, ret_features = False, ret_margins=False):

        preds = []
        device = self.device
        total_feature = []

        min_margins = []
        per_class_margins = []

        for i in tqdm(range(int(np.ceil(len(h_train) / 500.)))): 
            start = i * 500
            end = min((i + 1) * 500, len(h_train))
            idxs = list(range(start, end))
            
            h_batch = Batch.from_data_list([h_train[idx] for idx in idxs], follow_batch=['edge_index'])
            t_batch = Batch.from_data_list([t_train[idx] for idx in idxs], follow_batch=['edge_index'])
            #b_batch = Batch.from_data_list([b_train[idx] for idx in idxs], follow_batch=['edge_index'])
            
            rels_batch = [rels_train[idx] for idx in idxs]
            labels_batch = [labels_train[idx] for idx in idxs]
            rels_batch = torch.stack(rels_batch)
            labels_batch = torch.stack(labels_batch)
        
            #pred = self.model((h_batch.to(device), t_batch.to(device), rels_batch.to(device)))
            #preds.append(pred)
            if ret_features == True:
                pred, f = self.model((h_batch.to(device), t_batch.to(device), rels_batch.to(device)), ret_features = True)
                preds.append(pred)
                total_feature.append(f)
                
                if ret_margins == True:
                    min_margin, per_class_margin = self.compute_margins(pred, f)
                    min_margins.append(min_margin)
                    per_class_margins.append(per_class_margin)
            else:
                pred = self.model((h_batch.to(device), t_batch.to(device), rels_batch.to(device)))
                preds.append(pred)
            
        preds = torch.cat(preds, dim=0).squeeze(-1)
        if ret_features == True:
            f_tensor = torch.cat(total_feature, dim = 0).squeeze(-1)
            # BASE
            if ret_margins:
                min_margins = torch.cat(min_margins, dim=0)
                per_class_margins = torch.cat(per_class_margins, dim=0)
                return preds, f_tensor, min_margins, per_class_margins
            else:
                return preds, f_tensor
        else:
            return preds
    
    def test(self, h_test, t_test, rels_test, labels_test):
        device = self.device

        preds = []

        for i in tqdm(range(int(np.ceil(len(h_test) / 500.)))):

            start = i * 500
            end = min((i + 1) * 500, len(h_test))
            idxs = list(range(start, end))
            
            h_batch = Batch.from_data_list([h_test[idx] for idx in idxs], follow_batch=['edge_index'])
            t_batch = Batch.from_data_list([t_test[idx] for idx in idxs], follow_batch=['edge_index'])
            #b_batch = Batch.from_data_list([b_test[idx] for idx in idxs], follow_batch=['edge_index'])
            
            rels_batch = [rels_test[idx] for idx in idxs]
            labels_batch = [labels_test[idx] for idx in idxs]
            rels_batch = torch.stack(rels_batch)
            labels_batch = torch.stack(labels_batch)
            
            pred = self.model((h_batch.to(device), t_batch.to(device), rels_batch.to(device)))
            preds.append(pred)
        
            #probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
            #ground_truth.append(np.ones(len(p_score)))
            #test_probas_pred.append(probas_pred)
        preds = torch.cat(preds, dim=0).squeeze(-1)
        return preds

    def compute_margins(self, logits, embedding):
        
        weight = self.model.KGE.decoder[-1].weight
        bias = self.model.KGE.decoder[-1].bias
        
        predictions = logits.max(dim=1).indices

        # weight.shape = (C, M) C=number of classes, M = embedding size
        weight_orilogit = weight[predictions, :]
        # weight_orilogit.shape = (B, M)
        weight_delta = weight_orilogit[:, None, :] - weight[None, :]
        # weight_delta.shape = (B, C, M)
        # (B, 1, M) - (1, C, M)
        bias_delta = bias[predictions, None] - bias[None, :]
        # bias_delta.shape = (B, C)
        # (B, 1) - (1, C)
        lam_numerator = 2 * ((embedding[:, None, :] * weight_delta).sum(dim=2) + bias_delta)
        # (B, 1, M) * (B, C, M)
        # lam_numerator.shape = (B, C)
        lam_denominator = (weight_delta ** 2).sum(dim=2)
        # lam_denominator.shape = (B, C)
        lam = lam_numerator / lam_denominator
        epsilon = -weight_delta * lam[:, :, None] / 2
        # epsilon.shape = (B, C, M)
        radius = torch.linalg.norm(epsilon, dim=2)
        radius = torch.where(torch.isnan(radius).to(self.device), torch.tensor(float('inf')).to(self.device), radius)
        # radius.shape = (B, C)
        margins, min_margins_idx = radius.min(dim=1)

        return margins, radius