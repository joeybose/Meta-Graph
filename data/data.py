from itertools import product
import os
import os.path as osp
import json
import torch
import numpy as np
import ipdb
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader,DataListLoader
import ssl
from utils.utils import calculate_max_nodes_in_dataset, filter_dataset
import urllib
from random import shuffle
from torch_geometric.datasets import Planetoid,PPI,TUDataset
from .aminer_dataset import Aminer_Dataset

def load_dataset(name,args):
    ssl._create_default_https_context = ssl._create_unverified_context
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    args.fail_counter = 0
    args.resplit = True
    if name == 'PPI':
        train_dataset = PPI(path, split='train',transform=T.NormalizeFeatures())
        val_dataset = PPI(path, split='test',transform=T.NormalizeFeatures())
        test_dataset = PPI(path, split='test',transform=T.NormalizeFeatures())
        args.num_features = train_dataset.num_features
        max_nodes = calculate_max_nodes_in_dataset(train_dataset + val_dataset + test_dataset,\
                                                   args.min_nodes)
        total_graphs = len(train_dataset) + len(val_dataset) + len(test_dataset)
        print("Total Graphs in PPI %d" %(total_graphs))
    else:
        if name == 'ENZYMES':
            dataset = list(TUDataset(path,name,use_node_attr=True,\
                    transform=T.NormalizeFeatures()))
            shuffle(dataset)
        elif name =='REDDIT-MULTI-12K':
            dataset = list(TUDataset(path,name))
            shuffle(dataset)
            max_nodes = calculate_max_nodes_in_dataset(dataset,args.min_nodes)
            dataset = filter_dataset(dataset,args.min_nodes,max_nodes)
            args.feats = torch.randn(max_nodes,args.num_fixed_features,requires_grad=False)
            assert(args.use_fixed_feats or args.use_same_fixed_feats)
        elif name =='FIRSTMM_DB':
            dataset = list(TUDataset(path,name))
            shuffle(dataset)
            max_nodes = calculate_max_nodes_in_dataset(dataset,args.min_nodes)
        elif name =='DD':
            dataset = list(TUDataset(path,name))
            shuffle(dataset)
            max_nodes = calculate_max_nodes_in_dataset(dataset,args.min_nodes)
            dataset = filter_dataset(dataset,args.min_nodes,args.max_nodes)
        elif name =='AMINER':
            if args.opus:
                path = '/mnt/share/ankit.jain/meta-graph-data/AMINER/'
            dataset = Aminer_Dataset(path,name, args.k_core, \
                    reprocess=args.reprocess, ego=args.ego)
            dataset = [dataset[i] for i in range(0,len(dataset))]
            shuffle(dataset)
            max_nodes = calculate_max_nodes_in_dataset(dataset,args.min_nodes)
            dataset = filter_dataset(dataset,args.min_nodes,args.max_nodes)
        elif name == 'Cora':
            dataset = Planetoid(path, "Cora", T.NormalizeFeatures())
        else:
            raise NotImplementedError
        num_graphs = len(dataset)
        print("%d Graphs in Dataset" %(num_graphs))
        if num_graphs == 1:
            train_dataset = dataset
            val_dataset = dataset
            test_dataset = dataset
        else:
            train_cutoff = int(np.round(args.train_ratio*num_graphs))
            val_cutoff = train_cutoff + int(np.round(args.val_ratio*num_graphs))
            train_dataset = dataset[:train_cutoff]
            val_dataset = dataset[train_cutoff:val_cutoff]
            test_dataset = dataset[val_cutoff:]
        try:
            args.num_features = train_dataset[0].x.shape[1]
        except:
            ## TODO: Load Fixed Random Features
            print("Using Fixed Features")
            args.num_features = args.num_fixed_features
    if args.concat_fixed_feats:
        args.num_features = args.num_features + args.num_concat_features
    print("Node Features: %d" %(args.num_features))
    train_loader = DataListLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    val_loader = DataListLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataListLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader,val_loader,test_loader


