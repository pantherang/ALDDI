def training_config():
    config = {}
    config['dataset'] = 'db'
    config['num_rels'] = 86    
    # passive(random)  shannon(entropy)  most_likely_positive  k_center_greedy  BASE  MDC
    config['strategy'] = 'passive' 
    # SA-DDI GMPNN SSI-DDI DSN-DDI DGNN-DDI
    config['encoder'] = 'SA-DDI'    
    config['cuda_id'] = 1
    
    if config['encoder'] == 'DGNN-DDI':
        config['train_num_epoch'] = 50
        config['train_batch_size'] = 256 
    else:
        config['train_num_epoch'] = 200
        config['train_batch_size'] = 1024 
    
    config['init_batch_size'] = 5000
    config['query_batch_size'] = 5000
    
    config['cross_folds'] = True
    config['n_fold'] = 0 # 0 1 2 3 4 
    
    return config
