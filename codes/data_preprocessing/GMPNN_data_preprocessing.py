import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

# GMPNN
class PairData(Data):

    def __init__(self, j_indices, i_indices, pair_edge_index):
        super().__init__()
        self.i_indices = i_indices
        self.j_indices = j_indices
        self.edge_index = pair_edge_index
        self.num_nodes = None

    def __inc__(self, key, value, *args, **kwargs):
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'edge_index':
            return torch.tensor([[self.j_indices.shape[0]], [self.i_indices.shape[0]]])
        if key in ('i_indices', 'j_indices'):
            return 1
        return super().__inc__(key, value, *args, *kwargs)
            # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
            # Replace with "return super().__inc__(self, key, value, args, kwargs)"

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value, args, kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"

# _get_new_batch_id_and_num_nodes(h, old_id_to_new_batch_id, batch_drug_feats, all_drug_data, node_ind_seqs)
def _get_new_batch_id_and_num_nodes(old_id, old_id_to_new_batch_id, all_drug_data, node_ind_seqs):
    flag = 0
    new_id = old_id_to_new_batch_id.get(old_id, -1)
    num_nodes = all_drug_data[old_id].x.size(0)
    if new_id == - 1:
        flag = 1
        new_id = len(old_id_to_new_batch_id)
        old_id_to_new_batch_id[old_id] = new_id
        
        #batch_drug_feats.append(all_drug_data[old_id])
        start = (node_ind_seqs[-1][-1] + 1) if len(node_ind_seqs) else 0
        node_ind_seqs.append(torch.arange(num_nodes) + start)  
          
    return new_id, num_nodes, flag


# combo_idx = _get_combo_index((idx_h, idx_t), (h, t), already_in_combo, batch_unique_pairs, (h_num_nodes, t_num_nodes), node_j_ind_seqs_for_pair, node_i_ind_seqs_for_pair, node_ind_seqs, drug_num_node_indices, bipartite_edge_dict)
def _get_combo_index(combo, old_combo, already_in_combo, unique_pairs, num_nodes, node_j_ind_seqs_for_pair, node_i_ind_seqs_for_pair, node_ind_seqs, drug_num_node_indices, bipartite_edge_dict):
        
        idx = already_in_combo.get(combo, -1)
        if idx == -1:
            idx = len(already_in_combo)
            already_in_combo[combo] = idx
            
            pair_edge_index = bipartite_edge_dict.get(old_combo)
            if pair_edge_index is None:
                index_j = torch.arange(num_nodes[0]).repeat_interleave(num_nodes[1])
                index_i = torch.arange(num_nodes[1]).repeat(num_nodes[0])
                pair_edge_index = torch.stack([index_j, index_i])
                bipartite_edge_dict[old_combo] = pair_edge_index

            j_num_indices, i_num_indices = drug_num_node_indices[old_combo[0]], drug_num_node_indices[old_combo[1]]
            unique_pairs.append(PairData(j_num_indices, i_num_indices, pair_edge_index))
            node_j_ind_seqs_for_pair.append(node_ind_seqs[combo[0]])
            node_i_ind_seqs_for_pair.append(node_ind_seqs[combo[1]])
            
        return idx

def GMPNN_get_batch_data(h_id_batch, t_id_batch, rels_batch, all_drug_data):
        
    total_len = len(h_id_batch)
    # drug_id -> batch_id
    old_id_to_new_batch_id = {} 
    already_in_combo = {}
    bipartite_edge_dict = dict()
    
    batch_drug_feats = []
    node_ind_seqs = []
    
    node_i_ind_seqs_for_pair = []
    node_j_ind_seqs_for_pair = []
    batch_unique_pairs = [] 
    
    combo_indices = []
    rels = []
    
    all_drug_data = {drug_id: CustomData(x=data[0], edge_index=data[1], edge_feats=data[2], line_graph_edge_index=data[3])
                    for drug_id, data in all_drug_data.items()}

    ## To speed up training
    drug_num_node_indices = {
            drug_id: torch.zeros(data.x.size(0)).long() for drug_id, data in all_drug_data.items()
        }
    
    for i in range(total_len):
        h, t, r = h_id_batch[i], t_id_batch[i], rels_batch[i]

        idx_h, h_num_nodes, flag = _get_new_batch_id_and_num_nodes(h, old_id_to_new_batch_id, all_drug_data, node_ind_seqs)
        if flag:
            batch_drug_feats.append(all_drug_data[h])
        idx_t, t_num_nodes, flag = _get_new_batch_id_and_num_nodes(t, old_id_to_new_batch_id, all_drug_data, node_ind_seqs)
        if flag:
            batch_drug_feats.append(all_drug_data[t])

        combo_idx = _get_combo_index((idx_h, idx_t), (h, t), already_in_combo, batch_unique_pairs, (h_num_nodes, t_num_nodes), node_j_ind_seqs_for_pair, node_i_ind_seqs_for_pair, node_ind_seqs, drug_num_node_indices, bipartite_edge_dict)

        combo_indices.append(combo_idx)

        rels.append(int(r))

    batch_drug_data = Batch.from_data_list(batch_drug_feats, follow_batch=['edge_index'])
    
    batch_drug_pair_indices = torch.LongTensor(combo_indices) 

    batch_unique_drug_pair = Batch.from_data_list(batch_unique_pairs, follow_batch=['edge_index'])

    node_j_for_pairs = torch.cat(node_j_ind_seqs_for_pair)
    node_i_for_pairs = torch.cat(node_i_ind_seqs_for_pair)
    rels = torch.LongTensor(rels) 
    #for i in range(len(rels)):
    #    print(rels[i])
    return (batch_drug_data, batch_unique_drug_pair, rels, batch_drug_pair_indices, node_j_for_pairs, node_i_for_pairs)
