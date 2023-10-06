import os
import numpy as np
import random
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from torch_geometric.data import Batch 
import math 
from training_config import training_config
import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(4)
# related functions
from get_ddis import *
# encoder
# SADDI 
from SADDI.SADDI_trainer import Passive_SADDI_Trainer
from data_preprocessing.data_processing_saddi import CustomData 
from data_preprocessing.SADDI_data_preprocessing import *
# GMPNN 
from GMPNN.GMPNN_trainer import Passive_GMPNN_Trainer
from data_preprocessing.GMPNN_data_preprocessing import *
# DSNDDI 
from DSNDDI.DSNDDI_trainer import Passive_DSNDDI_Trainer
from data_preprocessing.DSNDDI_data_preprocessing import *
# SSIDDI 
from SSIDDI.SSIDDI_trainer import Passive_SSIDDI_Trainer
#from data_preprocessing.SSIDDI_data_preprocessing import * # in DSNDDI_data_preprocessing
# DGNNDDI
from DGNNDDI.DGNNDDI_trainer import Passive_DGNNDDI_Trainer
#from data_preprocessing.data_processing_saddi import CustomData 
from data_preprocessing.DGNN_data_preprocessing import * 

# strategy
from strategy.passive import *
from strategy.shannon import *
from strategy.most_likely_positive import *
from strategy.k_center_greedy import * # Core-Set 
from strategy.BASE import *
from strategy.MDC import *

def get_ddis(df, dataset):
    if dataset == 'db':
        return get_all_ddi_triplets_drugbank(df)

def main():
    #set_seed(100)
    ## seed
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)  
    np.random.seed(100)
    random.seed(100)
    os.environ['PYTHONHASHSEED'] = str(100)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    config = training_config() 
    print(config)
    dataset = config['dataset']
    num_rels = config['num_rels']
    strategy = config['strategy'] 
    encoder = config['encoder'] 
    cuda_id = config['cuda_id']
    device = torch.device(f'cuda:{cuda_id}') 
    train_num_epoch = config['train_num_epoch'] 
    train_batch_size = config['train_batch_size']
    init_batch_size = config['init_batch_size']
    query_batch_size = config['query_batch_size']
    
    cross_folds = config['cross_folds']
    n_fold = config['n_fold']

    #n_fold = 0 # 0 1 2 3 4
    training_path = f'./data/5-folds/train_fold_{n_fold}.csv'
    testing_path = f'./data/5-folds/test_fold_{n_fold}.csv'
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)
    
    h_id_train, t_id_train, rels_train, labels_train = get_ddis(train_df, dataset)
    h_id_test, t_id_test, rels_test, labels_test = get_ddis(test_df, dataset)
    
    num_iters = math.ceil( ( len(train_df) - init_batch_size ) / query_batch_size ) + 1
    print("Num of iters:{}".format(num_iters))
        
    if encoder == 'SA-DDI':
        trainer = Passive_SADDI_Trainer(device, train_batch_size = train_batch_size, train_num_epoch=train_num_epoch, num_rels = num_rels)
    elif encoder == 'GMPNN':
        trainer = Passive_GMPNN_Trainer(device, train_batch_size = train_batch_size, train_num_epoch=train_num_epoch, num_rels = num_rels)
    elif encoder == 'DSN-DDI':
        trainer = Passive_DSNDDI_Trainer(device, train_batch_size = train_batch_size, train_num_epoch=train_num_epoch, num_rels = num_rels)
    elif encoder == 'SSI-DDI':
        trainer = Passive_SSIDDI_Trainer(device, train_batch_size = train_batch_size, train_num_epoch=train_num_epoch, num_rels = num_rels)
    elif encoder == 'DGNN-DDI':
        trainer = Passive_DGNNDDI_Trainer(device, train_batch_size = train_batch_size, train_num_epoch=train_num_epoch, num_rels = num_rels)
    
    if encoder == 'SA-DDI':
        tri_train = SADDI_preprocess_data(h_id_train, t_id_train, rels_train, labels_train, dataset)
        tri_test = SADDI_preprocess_data(h_id_test, t_id_test, rels_test, labels_test, dataset)
    elif encoder == 'DGNN-DDI':
        tri_train = DGNN_preprocess_data(h_id_train, t_id_train, rels_train, labels_train, dataset)
        tri_test = DGNN_preprocess_data(h_id_test, t_id_test, rels_test, labels_test, dataset)
    elif encoder == 'GMPNN':
        tri_train = (h_id_train, t_id_train, rels_train, labels_train)
        tri_test = (h_id_test, t_id_test, rels_test, labels_test)
    elif encoder == 'DSN-DDI':
        tri_train =  DSNDDI_preprocess_data(h_id_train, t_id_train, rels_train, labels_train)
        tri_test = DSNDDI_preprocess_data(h_id_test, t_id_test, rels_test, labels_test)
    elif encoder == 'SSI-DDI':
        tri_train = SSIDDI_preprocess_data(h_id_train, t_id_train, rels_train, labels_train)
        tri_test = SSIDDI_preprocess_data(h_id_test, t_id_test, rels_test, labels_test)
    

    if strategy == 'passive':
        model = passive(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels)
    elif strategy == 'shannon': 
        model = shannon(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels)
    elif strategy == 'most_likely_positive': 
        model = most_likely_positive(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels)
    elif strategy == 'k_center_greedy': 
        model = k_center_greedy(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels)
    elif strategy == 'BASE': 
        model = BASE(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels)
    elif strategy == 'MDC': 
        model = MDC(tri_train, tri_test, encoder, dataset, trainer, init_batch_size, query_batch_size, num_iters, num_rels)

if __name__=='__main__':
    main()