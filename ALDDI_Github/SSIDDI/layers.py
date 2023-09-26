import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F

class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores

        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels # db 86
        self.n_features = n_features # 128
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # multi-class
        self.decoder = nn.Sequential(
            nn.Linear(4 * 4, 4 * 4),
            nn.PReLU(),
            nn.BatchNorm1d(4 * 4),
            nn.Linear(4 * 4, n_rels)
        )
    def forward(self, heads, tails, rels, alpha_scores, ret_features = False):
        #rels = self.rel_emb(rels)
        #rels = F.normalize(rels, dim=-1) 
        heads = F.normalize(heads, dim=-1) # [N, 4, 128]
        tails = F.normalize(tails, dim=-1) # [N, 4, 128]
        #rels = rels.view(-1, self.n_features, self.n_features) # [N, 128, 128]

        #scores = heads @ rels @ tails.transpose(-2, -1)
        scores = heads @ tails.transpose(-2, -1)
        
        if alpha_scores is not None:
          scores = alpha_scores * scores

        scores = scores.view(scores.size(0), -1)
        features = scores
        scores = self.decoder(scores)
        if ret_features == False:
            return scores
        else:
            return scores, features
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"

