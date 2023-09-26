import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn import metrics
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Set2Set,
                                )

from DSNDDI.layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    )
import time
from torch_geometric.data import Data, Batch
from torch.autograd import Variable

# heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2]
class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        
        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples, ret_features = False):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data,t_data,b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)
        
            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        # print(kge_heads.size(), kge_tails.size(), rels.size())
        attentions = self.co_attention(kge_heads, kge_tails)
        if ret_features == False:
            scores = self.KGE(kge_heads, kge_tails, rels, attentions)
            return scores
        else:
            scores, features = self.KGE(kge_heads, kge_tails, rels, attentions, ret_features)
            return scores, features
        '''
        # attentions = None
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores     
        '''
#intra+inter
class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        
        self.intraAtt = IntraGraphAttention(head_out_feats*n_heads)
        self.interAtt = InterGraphAttention(head_out_feats*n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
    
    def forward(self, h_data,t_data,b_graph):

        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)        
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)
        
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        
        h_interRep,t_interRep = self.interAtt(h_data,t_data,b_graph)
        
        h_rep = torch.cat([h_intraRep,h_interRep],1)
        t_rep = torch.cat([t_intraRep,t_interRep],1)
        
        h_data.x = h_rep
        t_data.x = t_rep

        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, batch=t_data.batch)
      
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)
        

        return h_data,t_data, h_global_graph_emb,t_global_graph_emb

class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature
    
    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights= F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
            
        len_p = p_scores.size()[0]
        p_loss = - F.logsigmoid(p_scores).mean()
        len_n = n_scores.size()[0]
        n_loss = - F.logsigmoid(-n_scores).mean()
        len_all = len_p + len_n
        return p_loss * (len_p/len_all) + n_loss * (len_n/len_all) , p_loss, n_loss 

# self.model = train_DSNDDI(h_train, t_train, rels_train, b_train, labels_train, train_batch_size=self.train_batch_size, train_num_epoch = self.train_num_epoch, device = self.device)
def train_DSNDDI(h_train, t_train, rels_train, b_train, labels_train, device, train_batch_size, train_num_epoch, num_rels):
    print("the len of training data this round is:{}".format(len(h_train)))
    num_of_batch = len(rels_train) // train_batch_size
    
    n_atom_feats = 55
    n_atom_hid = 128
    kge_dim = 128
    model = MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, num_rels, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    
    for iters in tqdm(range(train_num_epoch)):

        perm = torch.randperm(len(labels_train))
        for i in range(num_of_batch):
            start_idx = i * train_batch_size
            end_idx = min((i + 1) * train_batch_size, len(labels_train))

            idxs = perm[start_idx:end_idx]
            h_graphs_batch = [h_train[idx] for idx in idxs]
            t_graphs_batch = [t_train[idx] for idx in idxs]
            b_graphs_batch = [b_train[idx] for idx in idxs]
            
            h_graphs_batch = Batch.from_data_list(h_graphs_batch, follow_batch=['edge_index'])
            t_graphs_batch = Batch.from_data_list(t_graphs_batch, follow_batch=['edge_index'])
            b_graphs_batch = Batch.from_data_list(b_graphs_batch, follow_batch=['edge_index'])
            
            rels_batch = [rels_train[idx] for idx in idxs]
            labels_batch = [labels_train[idx] for idx in idxs]
            rels_batch = torch.stack(rels_batch)
            labels_batch = torch.stack(labels_batch)
            
            pred = model((h_graphs_batch.to(device), t_graphs_batch.to(device), rels_batch.to(device), b_graphs_batch.to(device)))

            truth_label = Variable(torch.from_numpy(np.array(rels_batch)).long()).to(device)
            loss = criterion(pred, truth_label)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        scheduler.step() 

    return model 
