import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool)

from SSIDDI.layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    RESCAL
                    )

from torch import optim
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from sklearn import metrics
from torch.autograd import Variable

class SSI_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features # 55
        self.hidd_dim = hidd_dim # 128
        self.rel_total = rel_total
        self.kge_dim = kge_dim # 128
        self.n_blocks = len(blocks_params)
        
        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = SSI_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

        self.rmodule = nn.Embedding(rel_total, hidd_dim) # 128
        self.lin = nn.Sequential(
            nn.Linear(hidd_dim * 2, hidd_dim * 2),
            nn.PReLU(),
            nn.Linear(hidd_dim * 2, hidd_dim),
        )
        #  multi-class
        self.decoder = nn.Sequential(
            nn.Linear(hidd_dim * 2, hidd_dim * 2),
            nn.PReLU(),
            nn.BatchNorm1d(hidd_dim * 2),
            nn.Linear(hidd_dim * 2, rel_total)
        )
        
    def forward(self, triples, ret_features = False):
        h_data, t_data, rels = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

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

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))
        
        repr_h = torch.stack(repr_h, dim=-2) 
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t # [batch_size, 4, 128]
        

        attentions = self.co_attention(kge_heads, kge_tails)
        #scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        #return scores
        if ret_features == False:
            scores = self.KGE(kge_heads, kge_tails, rels, attentions)
            return scores
        else:
            scores, features = self.KGE(kge_heads, kge_tails, rels, attentions, ret_features)
            return scores, features
        
        
class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
    
    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores= self.readout(data.x, data.edge_index, batch=data.batch)
        global_graph_emb = global_add_pool(att_x, att_batch)

        # data = max_pool_neighbor_x(data)
        return data, global_graph_emb

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
        #return (p_loss + n_loss) / 2, p_loss, n_loss 

# self.model = train_SSIDDI(h_train, t_train, rels_train, labels_train, train_batch_size=self.train_batch_size, train_num_epoch = self.train_num_epoch, device = self.device)
def train_SSIDDI(h_train, t_train, rels_train, labels_train, device, train_batch_size, train_num_epoch, num_rels):
    print("the len of training data this round is:{}".format(len(h_train)))
    num_of_batch = len(rels_train) // train_batch_size
    n_atom_feats = 55
    n_atom_hid = 128
    kge_dim = 128
    model = SSI_DDI(n_atom_feats, n_atom_hid, kge_dim, num_rels, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    
    for iters in tqdm(range(train_num_epoch)):
        perm = torch.randperm(len(labels_train))
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
            epoch_loss += loss.item() * len(rels_batch) 
        scheduler.step()
        
    return model 
