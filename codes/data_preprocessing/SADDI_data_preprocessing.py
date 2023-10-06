import torch
import pickle
import pandas as pd
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
def SADDI_preprocess_data(h_id, t_id, rels, labels, dataset):
    if dataset == 'db':
        drug_graph = read_pickle('./SADDI/drug_data_db.pkl')
        
    h_graphs = []
    t_graphs = []
    for drug_id in h_id:
        h_graphs.append(drug_graph.get(drug_id))
    for drug_id in t_id:
        t_graphs.append(drug_graph.get(drug_id))
    return (h_graphs, t_graphs, rels, labels)