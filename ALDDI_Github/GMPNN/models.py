import torch
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree
from torch_scatter import scatter
from GMPNN.layers import (CoAttentionLayerDrugBank)
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import pickle
from sklearn import metrics
from data_preprocessing.GMPNN_data_preprocessing import *
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.autograd import Variable
class GmpnnCSNetDrugBank(nn.Module):
    def __init__(self, in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout=0):

        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.rel_total = rel_total
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_hid_feats = hid_feats * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats), 
            CustomDropout(self.dropout),
        )

        self.propagation_layer = GmpnnBlock(edge_feats, self.hid_feats, self.n_iter, dropout) 

        self.i_pro = nn.Parameter(torch.zeros(self.snd_hid_feats , self.hid_feats))
        self.j_pro = nn.Parameter(torch.zeros(self.snd_hid_feats, self.hid_feats))
        self.bias = nn.Parameter(torch.zeros(self.hid_feats ))
        self.co_attention_layer = CoAttentionLayerDrugBank(self.snd_hid_feats)
        self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats)


        glorot(self.i_pro)
        glorot(self.j_pro)


        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_hid_feats, self.snd_hid_feats)
        )
        # multi-class
        self.decoder = nn.Sequential(
            nn.Linear(hid_feats, hid_feats),
            nn.PReLU(),
            nn.BatchNorm1d(hid_feats),
            nn.Linear(hid_feats, rel_total)
        )
    def forward(self, batch, ret_features = False):
        # (batch_drug_data, batch_unique_drug_pair, rels, batch_drug_pair_indices, node_j_for_pairs, node_i_for_pairs)
        drug_data, unique_drug_pair, rels, drug_pair_indices, node_j_for_pairs, node_i_for_pairs = batch
        
        drug_data.x = self.mlp(drug_data.x)
        new_feats = self.propagation_layer(drug_data)
        drug_data.x = new_feats
        
        x_j = drug_data.x[node_j_for_pairs]
        x_i = drug_data.x[node_i_for_pairs]
        
        attentions = self.co_attention_layer(x_j, x_i, unique_drug_pair)

        pair_repr = attentions.unsqueeze(-1) * ((x_i[unique_drug_pair.edge_index[1]] @ self.i_pro) * (x_j[unique_drug_pair.edge_index[0]] @ self.j_pro))
        
        x_i = x_j = None ## Just to free up some memory space
        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None 
        attentions = None

        pair_repr = scatter(pair_repr, unique_drug_pair.edge_index_batch, reduce='add', dim=0)[drug_pair_indices]
        # multi-class
        scores = self.decoder(pair_repr)
        #return scores
        if ret_features == False:
            return scores
        else:
            return scores, pair_repr

    def compute_score(self, pair_repr, rels):
        batch_size = len(rels)
        neg_n = (len(pair_repr) - batch_size) // batch_size  # I case of multiple negative samples per positive sample.
        rels = torch.cat([rels, torch.repeat_interleave(rels, neg_n, dim=0)], dim=0)
        rels = self.rel_embs(rels)
        scores = (pair_repr * rels).sum(-1)
        p_scores, n_scores = scores[:batch_size].unsqueeze(-1), scores[batch_size:].view(batch_size, -1, 1)

        return p_scores, n_scores

class GmpnnBlock(nn.Module):
    def __init__(self, edge_feats, n_feats, n_iter, dropout):
        super().__init__()
        self.n_feats = n_feats
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_n_feats = n_feats * 2

        self.w_i = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.w_j = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.a = nn.Parameter(torch.Tensor(1, self.n_feats))
        self.bias = nn.Parameter(torch.zeros(self.n_feats))

        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feats, self.n_feats)
        )

        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )

        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )

        glorot(self.w_i)
        glorot(self.w_j)
        glorot(self.a)

        self.sml_mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats)
        )
    
    def forward(self, data):

        edge_index = data.edge_index
        edge_feats = data.edge_feats
        edge_feats = self.edge_emb(edge_feats)

        deg = degree(edge_index[1], data.x.size(0), dtype=data.x.dtype)

        assert len(edge_index[0]) == len(edge_feats)
        alpha_i = (data.x @ self.w_i)
        alpha_j = (data.x @ self.w_j)
        alpha = alpha_i[edge_index[1]] + alpha_j[edge_index[0]] + self.bias
        alpha = self.sml_mlp(alpha)

        assert alpha.shape == edge_feats.shape
        alpha = (alpha* edge_feats).sum(-1)

        alpha = alpha / (deg[edge_index[0]])
        edge_weights = torch.sigmoid(alpha)

        assert len(edge_weights) == len(edge_index[0])
        edge_attr = data.x[edge_index[0]] * edge_weights.unsqueeze(-1)
        assert len(alpha) == len(edge_attr)
        
        out = edge_attr
        for _ in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + (out * edge_weights.unsqueeze(-1))

        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.mlp(x)

        return x

    def mlp(self, x): 
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2

        return x


class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = (lambda x: x ) if p == 0 else nn.Dropout(p)
    
    def forward(self, input):
        return self.dropout(input)

def train_GMPNN(h_train, t_train, rels_train, labels_train, device, train_batch_size, train_num_epoch, num_rels, all_drug_data):

    print("the len of training data this round is:{}".format(len(labels_train)))
    num_of_batch = len(labels_train) // train_batch_size
    
    # GMPNN
    hid_feats = 64
    rel_total = num_rels 
    n_iters = 10
    dropout = 0
    
    NUM_FEATURES, _, NUM_EDGE_FEATURES = next(iter(all_drug_data.values()))[:3] 
    NUM_FEATURES, NUM_EDGE_FEATURES = NUM_FEATURES.shape[1], NUM_EDGE_FEATURES.shape[1]

    model = GmpnnCSNetDrugBank(NUM_FEATURES, NUM_EDGE_FEATURES, hid_feats, num_rels, n_iters, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    
    for iters in tqdm(range(train_num_epoch)):
        model.train()
        perm = torch.randperm(len(labels_train))
        for i in range(num_of_batch):
            
            start_idx = i * train_batch_size
            end_idx = min((i + 1) * train_batch_size, len(labels_train))
            idxs = perm[start_idx:end_idx]

            h_id_batch = [h_train[idx] for idx in idxs]
            t_id_batch = [t_train[idx] for idx in idxs]
            
            rels_batch = [rels_train[idx] for idx in idxs]
            rels_batch = torch.stack(rels_batch)
            
            labels_batch = [labels_train[idx] for idx in idxs]
            labels_batch = torch.stack(labels_batch)
            
            # (batch_drug_data, batch_unique_drug_pair, rels, batch_drug_pair_indices, node_j_for_pairs, node_i_for_pairs)
            tri_batch = GMPNN_get_batch_data(h_id_batch, t_id_batch, rels_batch, all_drug_data)            
            
            moved_tuple = []
            for item in tri_batch:
                moved_item = item.to(device)
                moved_tuple.append(moved_item)  
                   
            pred = model(moved_tuple)
            truth_label = Variable(torch.from_numpy(np.array(rels_batch)).long()).to(device)
            loss = criterion(pred, truth_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step() 
        
    return model 