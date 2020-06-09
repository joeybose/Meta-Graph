import torch
import os
import os.path as osp
import argparse
import ipdb
from collections import OrderedDict
import torch
import math
from statistics import median, mean
import random
import numpy as np
import networkx as nx
import copy
from gensim.models import Word2Vec
from torch_geometric.utils import to_networkx
from networkx.generators.random_graphs import erdos_renyi_graph
from torch._six import inf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import grakel
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
import torch.nn as nn
from grakel import GraphKernel

''' Set Random Seed '''
def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not int(0), parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)
    return total_norm

def filter_state_dict(state_dict,name):
    keys_to_del = []
    for key in state_dict.keys():
        if name not in key:
            keys_to_del.append(key)
    for key in sorted(keys_to_del, reverse=True):
        del state_dict[key]
    return state_dict

'''Monitor Norm of gradients'''
def monitor_grad_norm(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

'''Monitor Norm of gradients'''
def monitor_grad_norm_2(gradients):
    total_norm = 0
    for p in gradients:
        if p is not int(0):
            param_norm = p.data.norm(2)
            total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

'''Monitor Norm of weights'''
def monitor_weight_norm(model):
    parameters = list(filter(lambda p: p is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index,weights):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index,weights)
    return model.test(z, pos_edge_index, neg_edge_index)

def global_test(args, model, data_batch, weights):
    model.eval()
    auc_list, ap_list = [], []
    for data in data_batch:
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data.batch = None
        # Test Ratio is Fixed at 0.1
        meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
        if args.use_fixed_feats and args.dataset=='REDDIT-MULTI-12K':
            ##TODO: Should this be a fixed embedding table instead of generating this each time?
            num_nodes = data.num_nodes
            perm = torch.randperm(args.feats.size(0))
            perm_idx = perm[:num_nodes]
            data.x = args.feats[perm_idx]
        elif args.use_same_fixed_feats and args.dataset=='REDDIT-MULTI-12K':
            node_feats = args.feats[0].unsqueeze(0).repeat(num_nodes,1)
            data.x = node_feats
        try:
            x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
        except:
            data = model.split_edges(data,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
            x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
        try:
            pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        except:
            print("Failed in Global Test")
            args.fail_counter += 1
            continue

        # Additional Failure Checks for small graphs
        if pos_edge_index.size()[1] == 0 or neg_edge_index.size()[1] == 0:
            args.fail_counter += 1
            print("Failed on Graph")
            continue

        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index,weights)
        auc, ap = model.test(z, pos_edge_index, neg_edge_index)
        auc_list.append(auc)
        ap_list.append(ap)
    return auc_list, ap_list

def filter_dataset(dataset,min_nodes,max_nodes):
    filtered_dataset = []
    for i, graph in enumerate(dataset):
        num_nodes = graph.num_nodes
        if num_nodes >= min_nodes and num_nodes <= max_nodes:
            filtered_dataset.append(graph)
    return filtered_dataset

def run_analysis(args, meta_model, train_loader):
    degree_scores = avg_node_deg(args, meta_model, train_loader)
    print("Finished Computing Avg node degree difference")
    diff_num_nodes_scores = diff_num_nodes(args, meta_model, train_loader)
    print("Finished Computing node number difference")
    diff_num_edges_scores = diff_num_edges(args, meta_model, train_loader)
    print("Finished Computing edge number difference")
    diff_avg_clustering_coeff_scores = diff_avg_clustering_coeff(args, meta_model, train_loader)
    print("Finished Computing Avg clustering coefficient difference")
    diff_avg_triangles_scores = diff_avg_triangles(args, meta_model, train_loader)
    print("Finished Computing Avg triangles difference")
    emb_scores = avg_emb_cosine_mat(args, train_loader)
    print("Finished Computing Embedding Scores")
    sig_scores = compute_sig_graph_sim(args, meta_model, train_loader)
    print("Finished Sig Scores")
    wl_scores, wl_scores_neg = run_wl_kernel(args,train_loader,meta_model,\
            args.meta_train_edge_ratio)
    print("Finished WL Scores")
    try:
        deg_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
                degree_scores.reshape(-1,1))
        deg_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
                degree_scores.reshape(-1))
        print("Deg Spearman score %f p-value %f"%(deg_spearman_score[0],deg_spearman_score[1]))
        print("Deg Pearson score %f p-value %f"%(deg_pearson_score[0],deg_pearson_score[1]))
    except:
        print("Failed Degree Sim Test")

    emb_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            emb_scores.cpu().numpy().reshape(-1,1))
    emb_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            emb_scores.cpu().numpy().reshape(-1))

    spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            wl_scores.reshape(-1,1))
    pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            wl_scores.reshape(-1))

    spearman_score_neg = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            wl_scores_neg.reshape(-1,1))
    pearson_score_neg = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            wl_scores_neg.reshape(-1))

    diff_num_nodes_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            diff_num_nodes_scores.reshape(-1,1))
    diff_num_nodes_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            diff_num_nodes_scores.reshape(-1))

    diff_num_edges_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1, 1), \
            diff_num_edges_scores.reshape(-1, 1))
    diff_num_edges_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1), \
            diff_num_edges_scores.reshape(-1))

    diff_avg_clustering_coeff_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1, 1), \
            diff_avg_clustering_coeff_scores.reshape(-1, 1))
    diff_avg_clustering_coeff_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1), \
            diff_avg_clustering_coeff_scores.reshape(-1))

    diff_avg_triangles_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1, 1), \
            diff_avg_triangles_scores.reshape(-1, 1))
    diff_avg_triangles_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1), \
            diff_avg_triangles_scores.reshape(-1))

    print("Emb Spearman score %f p-value %f"%(emb_spearman_score[0],emb_spearman_score[1]))
    print("Emb Pearson score %f p-value %f"%(emb_pearson_score[0],emb_pearson_score[1]))
    print("Pos Spearman score %f p-value %f"%(spearman_score[0],spearman_score[1]))
    print("Pos Pearson score %f p-value %f"%(pearson_score[0],pearson_score[1]))
    print("Neg Spearman score %f p-value %f"%(spearman_score_neg[0],spearman_score_neg[1]))
    print("Neg Pearson score %f p-value %f"%(pearson_score_neg[0],pearson_score_neg[1]))
    print("Diff Num Nodes Spearman score %f p-value %f" % \
            (diff_num_nodes_spearman_score[0], diff_num_nodes_spearman_score[1]))
    print("Diff Num Nodes Pearson score %f p-value %f" % \
            (diff_num_nodes_pearson_score[0], diff_num_nodes_pearson_score[1]))
    print("Diff Num Edges Spearman score %f p-value %f" % \
            (diff_num_edges_spearman_score[0], diff_num_edges_spearman_score[1]))
    print("Diff Num Edges Pearson score %f p-value %f" % \
            (diff_num_edges_pearson_score[0], diff_num_edges_pearson_score[1]))
    print("Diff Avg Clustering Coeff Spearman score %f p-value %f" % \
            (diff_avg_clustering_coeff_spearman_score[0], diff_avg_clustering_coeff_spearman_score[1]))
    print("Diff Avg Clustering Coeff Pearson score %f p-value %f" % \
            (diff_avg_clustering_coeff_pearson_score[0], diff_avg_clustering_coeff_pearson_score[1]))
    print("Diff Avg Triangles Spearman score %f p-value %f" % \
            (diff_avg_triangles_spearman_score[0], diff_avg_triangles_spearman_score[1]))
    print("Diff Avg Triangles Pearson score %f p-value %f" % \
            (diff_avg_triangles_pearson_score[0], diff_avg_triangles_pearson_score[1]))

def calculate_max_nodes_in_dataset(dataset,min_nodes):
    max_nodes = 0
    graph_id = 0
    total_nodes = 0
    big_node_graphs = 0
    total_edges = 0
    for i, graph in enumerate(dataset):
        num_nodes = graph.num_nodes
        if num_nodes >= min_nodes:
            total_nodes += num_nodes
            total_edges += graph.edge_index.shape[1]
            big_node_graphs += 1
        if num_nodes > max_nodes:
            max_nodes = num_nodes
            graph_id = i
    avg_nodes = total_nodes / len(dataset)
    avg_edges = total_edges / len(dataset)
    print("Max nodes is %d in graph %d" %(max_nodes,graph_id))
    print("Avg nodes is %d" %(avg_nodes))
    print("Avg Edges is %d" %(avg_edges))
    print("Num 1000 nodes is %d| Total Graphs %d" %(big_node_graphs, len(dataset)))
    return max_nodes

def calc_adamic_adar_score(G, pos_edge_index, neg_edge_index):
    pos_list = pos_edge_index.t().detach().cpu().tolist()
    neg_list = neg_edge_index.t().detach().cpu().tolist()
    pos_preds, neg_preds = [],[]
    for pos_edge in pos_list:
        try:
            pos_score = list(nx.adamic_adar_index(G,[tuple(pos_edge)]))[0][2]
        except:
            pos_score = 0
        pos_preds.append(pos_score)

    for neg_edge in neg_list:
        try:
            neg_score = list(nx.adamic_adar_index(G,[tuple(neg_edge)]))[0][2]
        except:
            neg_score = 0
        neg_preds.append(neg_score)
    preds = list(pos_preds) + list(neg_preds)
    pos_y = pos_edge_index.new_ones(pos_edge_index.size(1))
    neg_y = neg_edge_index.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0).detach().cpu().numpy()
    return roc_auc_score(y, preds), average_precision_score(y, preds)

def calc_deepwalk_score(pos_edge_index, neg_edge_index,node_vectors,entity2index):
    '''

    :param G:
    :param pos_edge_index:
    :param neg_edge_index:
    :param node_vectors: node vectors from deepwalk model
    :return:
    '''
    pos_list = pos_edge_index.t().detach().cpu().tolist()
    neg_list = neg_edge_index.t().detach().cpu().tolist()
    pos_preds, neg_preds = [], []
    for pos_edge in pos_list:
        node1, node2= entity2index[pos_edge[0]], entity2index[pos_edge[1]]
        pos_score = cosine_similarity(node_vectors[node1].reshape(1,-1),\
                node_vectors[node2].reshape(1,-1))[0]
        # Scaling the cosine similarity between 0 to 1
        # pos_score = (pos_score +1) /2
        pos_preds.append(pos_score)

    for neg_edge in neg_list:
        node1, node2 = entity2index[neg_edge[0]], entity2index[neg_edge[1]]
        neg_score = cosine_similarity(node_vectors[node1].reshape(1,-1),\
                node_vectors[node2].reshape(1,-1))[0]
        # Scaling the cosine similarity between 0 to 1
        # neg_score = (neg_score + 1) / 2
        neg_preds.append(neg_score)
    preds = list(pos_preds) + list(neg_preds)
    pos_y = pos_edge_index.new_ones(pos_edge_index.size(1))
    neg_y = neg_edge_index.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0).detach().cpu().numpy()
    return roc_auc_score(y, preds), average_precision_score(y, preds)

def do_random_walks(G,path_length, alpha=0, rand=random.Random(), start=None):

  if start!= None:
      path=[start]
  else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.nodes()))]

  while len(path) < path_length:
    cur = path[-1]
    num_neighbors= [n for n in G.neighbors(cur)]
    if len(num_neighbors) > 0:
        if rand.random() >= alpha:
            path.append(rand.choice(num_neighbors))
        else:
            path.append(path[0])
    else:
        break
  return [str(node) for node in path]


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(do_random_walks(G,path_length, rand=rand, alpha=alpha, start=node))

    return walks

def train_deepwalk_model(G,number_walks = 10, walk_length = 80,seed= 0,epochs =1):
    '''
    Trains the deepwalk model for baseline puporses on the test set.
    :param edge_array:
    :return:
    '''
    # Random Walk Generation
    walks = build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    print("Training...")
    model = Word2Vec(walks, size=64, window=5, min_count=0, sg=1, hs=1, workers=5,iter=epochs)
    node_vectors = model.wv.vectors
    index2entity = model.wv.index2entity
    entity2index = {}
    for i in range(len(index2entity)):
        entity2index[int(index2entity[i])] = i
    return node_vectors,entity2index,index2entity


def create_nx_graph(data):
    edges_train = data.train_pos_edge_index.t().detach().cpu().tolist()
    G_train = nx.Graph()
    G_train.add_edges_from(edges_train)
    return G_train

def create_nx_graph_deepwalk(data):
    edges_test = data.test_pos_edge_index.t().detach().cpu().tolist()
    edges_train = data.train_pos_edge_index.t().detach().cpu().tolist()
    edges_val = data.val_pos_edge_index.t().detach().cpu().tolist()
    all_pos_edges = edges_train + edges_test + edges_val
    #Getting the test negative edges as we will evaluate deepwalk on that.
    edges_test_neg = data.test_neg_edge_index.t().detach().cpu().tolist()
    all_neg_edges_arr = np.array(edges_test_neg)

    G = nx.Graph()
    edges_arr= np.array(all_pos_edges)
    # Taking care of isolated nodes in the graph
    max_node_id = max(max(edges_arr[:,0]), max(edges_arr[:,1]))
    # Taking the max node index of test negative edges if present
    max_node_id = max(max_node_id, max(all_neg_edges_arr[:,0]), max(all_neg_edges_arr[:,1]))
    #Add nodes here
    G.add_nodes_from([x for x in range(max_node_id+1)])
    G.add_edges_from(edges_train)
    return G

def subsample_edges(G, num_edges):
    total_edges = G.edge_index.shape()[1]
    perm = torch.randperm(G.edge_index)
    subsampled_edges = perm[:num_edges]
    G.edge_index = subsampled_edges
    return G

def val(model, args, x, val_pos_edge_index, num_nodes, weights):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, val_pos_edge_index, weights)
        loss = model.recon_loss(z, val_pos_edge_index)
        if args.model in ['VGAE']:
            loss = loss + (1 / num_nodes) * model.kl_loss()
    return loss.item()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def create_masked_networkx_graph(data):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph`.
    Args:
        data (torch_geometric.data.Data): The data object.
    """

    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))

    values = {key: data[key].squeeze().tolist() for key in data.keys}

    for i, (u, v) in enumerate(data.train_pos_edge_index.t().tolist()):
        G.add_edge(u, v)
    return G

def avg_node_deg(args, model, dataloader):
    dataset = dataloader.dataset
    node_degree_list = []
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    for data_graph in dataset:
        sum_node_degree = 0
        data = model.split_edges(data_graph,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        graph_degree_list = []
        for node_id in nx_graph.nodes:
            sum_node_degree += nx_graph.degree[node_id]
            graph_degree_list.append(nx_graph.degree[node_id])
        log_median_node_degree = np.log(median(graph_degree_list))
        log_avg_node_degree = np.log(sum_node_degree / len(nx_graph))
        node_degree_list.append(log_median_node_degree)

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(node_degree_list[i] - node_degree_list[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_num_nodes(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    num_nodes_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev), \
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        num_nodes_per_dataset.append(len(nx_graph.nodes))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(num_nodes_per_dataset[i] - num_nodes_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_num_edges(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    num_edges_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        num_edges_per_dataset.append(len(nx_graph.edges))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(num_edges_per_dataset[i] - num_edges_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_avg_clustering_coeff(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    avg_clustering_coeff_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        avg_clustering_coeff_per_dataset.append(nx.average_clustering(nx_graph))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(avg_clustering_coeff_per_dataset[i] - avg_clustering_coeff_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_avg_triangles(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    avg_triangles_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        avg_triangles_per_dataset.append(mean(nx.triangles(nx_graph).values()))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = abs(avg_triangles_per_dataset[i] - avg_triangles_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def avg_emb_cosine_mat(args, dataloader):
    dataset = dataloader.dataset
    embedding_list = []
    for data_graph in dataset:
        x = data_graph.x.to(args.dev)
        avg_emb = torch.mean(x,dim=0)
        norm = avg_emb.norm(p=2, dim=0, keepdim=True)
        norm_avg_emb = avg_emb.div(norm.expand_as(avg_emb))
        embedding_list.append(norm_avg_emb)
    avg_emb_mat = torch.stack(embedding_list)
    scores_mat = torch.mm(avg_emb_mat, avg_emb_mat.t())
    print("Avg Node Feature similarity between graphs: %f"%(scores_mat.mean()))
    return scores_mat

def compute_sig_graph_sim(args, model, dataloader):
    dataset = dataloader.dataset
    sig_params_list = []
    for data_graph in dataset:
        data_graph.train_mask = data_graph.val_mask = data_graph.test_mask = data_graph.y = None
        data_graph.batch = None
        num_nodes = data_graph.num_nodes
        # Val Ratio is Fixed at 0.2
        if args.concat_fixed_feats:
            if data_graph.x.shape[1] < args.num_features:
                concat_feats = torch.randn(num_nodes,args.num_concat_features,requires_grad=False)
                data_graph.x = torch.cat((data_graph.x,concat_feats),1)
        meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
        data_graph.x.cuda()
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)

        x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
        test_pos_edge_index, test_neg_edge_index = data.test_pos_edge_index.to(args.dev),\
                data.test_neg_edge_index.to(args.dev)
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index, \
                    OrderedDict(model.named_parameters()), inner_loop=True)
        sig_out = model.encoder.cache_sig_out
        flattened_params = torch.cat(sig_out)
        norm = flattened_params.norm(p=2, dim=0, keepdim=True)
        norm_sig_out = flattened_params.div(norm.expand_as(flattened_params))
        sig_params_list.append(norm_sig_out)
    sig_params_mat = torch.stack(sig_params_list)
    scores_mat = torch.mm(sig_params_mat, sig_params_mat.t())
    return scores_mat

def create_erdos_renyi_graph(num_nodes,edge_prob):
    graph = erdos_renyi_graph(num_nodes,edge_prob)
    return graph

def run_wl_kernel(args, dataloader, model, edge_ratio):
    dataset = dataloader.dataset
    pos_graph_list = []
    neg_graph_list = []
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    i = 0
    for graph in dataset:
        print("Graph : %d" %(i))
        i +=1
        try:
            x, train_pos_edge_index = graph.x.to(args.dev), \
                graph.train_pos_edge_index.to(args.dev)
            data = graph
        except:
            data = model.split_edges(graph,val_ratio=args.meta_val_edge_ratio,\
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        neg_graph = erdos_renyi_graph(len(nx_graph),edge_ratio)
        pos_edge_list = list(nx_graph.edges())
        neg_edge_list = list(neg_graph.edges())
        pos_node_dict = {}
        neg_node_dict = {}

        ''' Pos Samples '''
        for node_id in nx_graph.nodes:
            nx_graph.node[node_id]['label'] = nx_graph.degree[node_id]
            pos_node_dict[node_id] = nx_graph.degree[node_id]
            pos_grakel_graph = [pos_edge_list, pos_node_dict]
        pos_graph_list.append(pos_grakel_graph)

        '''Neg Samples '''
        for node_id in nx_graph.nodes:
            nx_graph.node[node_id]['label'] = nx_graph.degree[node_id]
            neg_node_dict[node_id] = nx_graph.degree[node_id]
            neg_grakel_graph = [neg_edge_list, neg_node_dict]
        neg_graph_list.append(neg_grakel_graph)
    wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "n_iter": 5},\
                {"name": "subtree_wl"}], Nystroem=len(dataset))
    kernel_mat = wl_kernel.fit_transform(pos_graph_list)
    neg_kernel_mat = wl_kernel.transform(neg_graph_list)
    return kernel_mat, neg_kernel_mat
