import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import math
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.nn import  global_add_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import  softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.data import Batch
from torch.autograd import Variable
class GlobalAttentionPool(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx

class DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)
    
        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):

        edge_index = data.edge_index
        # Recall that we have converted the node graph to the line graph, 
        # so we should assign each bond a bond-level feature vector at the beginning (i.e., h_{ij}^{(0)}) in the paper).
        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr
        
        # The codes below show the graph convolution and substructure attention.
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            # Lines 61 and 62 are the main steps of graph convolution.
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + out
            # Equation (1) in the paper
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        # Substructure attention, Equation (3)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        # Substructure attention, Equation (4),
        # Suppose batch_size=64 and iteraction_numbers=10. 
        # Then the scores will have a shape of (64, 1, 10), 
        # which means that each graph has 10 scores where the n-th score represents the importance of substructure with radius n.
        scores = torch.softmax(scores, dim=-1)
        # We should spread each score to every line in the line graph.
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)
        # Weighted sum of bond-level hidden features across all steps, Equation (5).
        out = (out_all * scores).sum(-1)
        # Return to node-level hidden features, Equations (6)-(7).
        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x

class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)

        return x   

class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)

        return x

class SA_DDI(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, num_rels, hidden_dim=64, n_iter=10):
        super(SA_DDI, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter)
        self.h_gpool = GlobalAttentionPool(hidden_dim)
        self.t_gpool = GlobalAttentionPool(hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.rmodule = nn.Embedding(num_rels, hidden_dim)

        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)

        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)
        # multi-class
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_rels)
        )
        
    def forward(self, triples, ret_features = False):
        h_data, t_data, rels = triples

        x_h = self.drug_encoder(h_data)
        x_t = self.drug_encoder(t_data)

        # Start of SSIM
        # TAGP, Equation (8)
        g_h = self.h_gpool(x_h, h_data.edge_index, h_data.batch)
        g_t = self.t_gpool(x_t, t_data.edge_index, t_data.batch)

        g_h_align = g_h.repeat_interleave(degree(t_data.batch, dtype=t_data.batch.dtype), dim=0)
        g_t_align = g_t.repeat_interleave(degree(h_data.batch, dtype=h_data.batch.dtype), dim=0)

        # Equation (10)
        h_scores = (self.w_i(x_h) * self.prj_i(g_t_align)).sum(-1)
        h_scores = softmax(h_scores, h_data.batch, dim=0)
        # Equation (10)
        t_scores = (self.w_j(x_t) * self.prj_j(g_h_align)).sum(-1)
        t_scores = softmax(t_scores, t_data.batch, dim=0)
        # Equation (11)
        h_final = global_add_pool(x_h * g_t_align * h_scores.unsqueeze(-1), h_data.batch)
        t_final = global_add_pool(x_t * g_h_align * t_scores.unsqueeze(-1), t_data.batch)
        # End of SSIM

        # multi-class
        pair = torch.cat([h_final, t_final], dim=-1)
        score = self.decoder(pair)
        if ret_features == False:
            return score
        else:
            return score, pair
        '''
        pair = torch.cat([h_final, t_final], dim=-1)
        rfeat = self.rmodule(rels)
        logit = (self.lin(pair) * rfeat).sum(-1)

        return logit
        '''
    
# self.model = train_SADDI(h_train, t_train, rels_train, labels_train, num_rels = self.num_rels, train_batch_size=self.train_batch_size, train_num_epoch = self.train_num_epoch, device = self.device)
def train_SADDI(h_train, t_train, rels_train, labels_train, device, train_batch_size, train_num_epoch, num_rels):

    print("the len of training data this round is:{}".format(len(h_train)))
    num_of_batch = len(rels_train) // train_batch_size

    node_dim = h_train[0].x.size(-1)
    edge_dim = h_train[0].edge_attr.size(-1)
    model = SA_DDI(node_dim, edge_dim, num_rels, n_iter = 10).to(device) 

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    
    for iters in tqdm(range(train_num_epoch)):

        perm = torch.randperm(len(labels_train))#.cuda()
        epoch_loss = 0
        for i in range(num_of_batch):
            start_idx = i * train_batch_size
            end_idx = min((i + 1) * train_batch_size, len(labels_train))
            idxs = perm[start_idx:end_idx]
            #idxs = perm[i * batch_size: min((i + 1) * batch_size, len(train_labels))]

            h_graphs_batch = [h_train[idx] for idx in idxs]
            t_graphs_batch = [t_train[idx] for idx in idxs]
            h_graphs_batch = Batch.from_data_list(h_graphs_batch, follow_batch=['edge_index'])
            t_graphs_batch = Batch.from_data_list(t_graphs_batch, follow_batch=['edge_index'])
            
            rels_batch = [rels_train[idx] for idx in idxs]
            labels_batch = [labels_train[idx] for idx in idxs]
            rels_batch = torch.stack(rels_batch)
            labels_batch = torch.stack(labels_batch)
            
            pred = model((h_graphs_batch.to(device), t_graphs_batch.to(device), rels_batch.to(device)))

            truth_label = Variable(torch.from_numpy(np.array(rels_batch)).long()).to(device)

            loss = criterion(pred, truth_label)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        scheduler.step() 
       
    return model 