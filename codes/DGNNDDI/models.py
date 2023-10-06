import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import  softmax
from torch_scatter import scatter
from torch.nn.modules.container import ModuleList
from torch_geometric.utils import degree
from torch_geometric.nn import LayerNorm, global_add_pool
import torch.optim as optim
#from layers import  CoAttentionLayer, RESCAL
from DGNNDDI.layers import *
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.autograd import Variable
import numpy as np

class GlobalAttentionPool(nn.Module):

    def __init__(self, hidden_dim):
        super(GlobalAttentionPool,self).__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx

class DMPNN(nn.Module):
    def __init__(self,  n_feats, n_iter):
        super(DMPNN,self).__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(n_feats, n_feats, bias=False)
    
        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):

        edge_index = data.edge_index

        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr
        

        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            # Lines 61 and 62 are the main steps of graph convolution.
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + out

            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)

        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias

        scores = torch.softmax(scores, dim=-1)

        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)

        out = (out_all * scores).sum(-1)

        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x

class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super(LinearBlock,self).__init__()
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
    def __init__(self,  hidden_dim, n_iter):
        super(DrugEncoder,self).__init__()
        self.line_graph = DMPNN( hidden_dim, n_iter)

    def forward(self, data):
        x = self.line_graph(data)

        return x


class MPNN_Block(nn.Module):
    def __init__(self, hidden_dim, n_iter):
        super(MPNN_Block,self).__init__()

        self.drug_encoder = DrugEncoder( hidden_dim, n_iter)
        self.readout = GlobalAttentionPool(hidden_dim)

    def forward(self, data):
        data.x = self.drug_encoder(data)
        global_graph_emb = self.readout(data.x, data.edge_index, data.batch)
        return data, global_graph_emb



class MPNN_DDI(nn.Module):
    def __init__(self,in_dim,edge_dim, hidden_dim, n_iter, kge_dim, rel_total):
        super().__init__()
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = 3
        self. lin_edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim))
        self.blocks = []
        self.net_norms = ModuleList()
        for i in range(self.n_blocks):
            block = MPNN_Block(hidden_dim,n_iter=n_iter)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples, ret_features = False):
        h_data, t_data, rels = triples
        h_data.x= self.mlp(h_data.x)
        t_data.x = self.mlp(t_data.x)

        h_data.edge_attr = self.lin_edge(h_data.edge_attr)
        t_data.edge_attr = self.lin_edge(t_data.edge_attr)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out1, out2 = block(h_data), block(t_data)

            h_data = out1[0]
            t_data = out2[0]
            r_h = out1[1]
            r_t = out2[1]

            repr_h.append(r_h)
            repr_t.append(r_t)

        repr_h = torch.stack(repr_h, dim=-2)      
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t    

        attentions = self.co_attention(kge_heads, kge_tails)    
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        if ret_features == False:
            scores = self.KGE(kge_heads, kge_tails, rels, attentions)
            return scores
        else:
            scores, features = self.KGE(kge_heads, kge_tails, rels, attentions, ret_features)
            return scores, features

        #return scores

def train_DGNN(h_train, t_train, rels_train, labels_train, device, train_batch_size, train_num_epoch, num_rels):

    print("the len of training data this round is:{}".format(len(h_train)))
    num_of_batch = len(rels_train) // train_batch_size
    node_dim = h_train[0].x.size(-1) 
    edge_dim = h_train[0].edge_attr.size(-1) 
    
    model = MPNN_DDI(node_dim,edge_dim,hidden_dim=64,n_iter=3, kge_dim=64, rel_total=86).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
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
            #pred = model(head_drug_mol_graph[idxs].cuda(), tail_drug_mol_graph[idxs].cuda(), train_rels[idxs].cuda())
            #loss = criterion(pred.to(device), labels_batch.to(device))
            loss = criterion(pred, truth_label)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        scheduler.step() 
        
    return model 
