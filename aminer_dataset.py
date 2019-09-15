import os
import os.path as osp
import shutil
from os import listdir
from os.path import isfile, join
import ipdb
from tqdm import tqdm
from operator import itemgetter
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url, extract_zip
import networkx as nx
import numpy as np
from torch_geometric.data.dataset import files_exist
from torch_geometric.utils import remove_self_loops

class Aminer_Dataset(Dataset):
    def __init__(self, root, name, k_core, reprocess=False, ego=False, transform=None, pre_transform=None):
        self.name = name
        self.raw_graphs_path = join(root,'raw')
        self.k_core = k_core
        self.reprocess = reprocess
        self.ego = ego
        print("Path is %s" %(self.raw_graphs_path))
        onlyfiles = [f for f in listdir(self.raw_graphs_path) if isfile(join(self.raw_graphs_path, f))]
        self.raw_files = onlyfiles
        super(Aminer_Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        if files_exist(self.processed_paths):  # pragma: no cover
            return
        onlyfiles = [f for f in listdir(self.raw_graphs_path) if isfile(join(self.raw_graphs_path, f))]
        self.raw_files = onlyfiles
        return onlyfiles

    @property
    def processed_file_names(self):
        onlyfiles = [f for f in listdir(self.processed_dir) if isfile(join(self.processed_dir, f))]
        return onlyfiles

    def __len__(self):
        return len(self.processed_file_names)

    def _process(self):
        if self.reprocess:
            self.process()
        if files_exist(self.processed_paths):  # pragma: no cover
            return
        print('Processing...')
        makedirs(self.processed_dir)
        self.process()
        print('Done!')

    def _download(self):
        pass

    def download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)

        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        print("Manually copy *.pkl files to AMINER/raw dir")

    def process(self):
        i = 0
        print("Path is %s" %(self.raw_graphs_path))
        onlyfiles = [f for f in listdir(self.raw_graphs_path) if isfile(join(self.raw_graphs_path, f))]
        self.raw_files = onlyfiles
        G_len_list = []
        print("Beginning to Process AMINER graphs")
        for j,raw_path in tqdm(enumerate(self.raw_files),total=len(self.raw_files)):
            path = path = osp.join(self.raw_dir, raw_path)
            # Read data from `raw_path`.
            G = nx.read_gpickle(path)
            # Compute K-core of Graph
            G = nx.k_core(G, k=self.k_core)
            # Re-index nodes from 0 to len(G)
            mapping = dict(zip(G, range(0, len(G))))
            G = nx.relabel_nodes(G, mapping)
            if self.ego:
                # Randomly sample node
                node_and_degree = G.degree()
                rand_node = np.random.choice(len(G),1)
                G = nx.ego_graph(G,rand_node[0],radius=2)
                mapping = dict(zip(G, range(0, len(G))))
                G = nx.relabel_nodes(G, mapping)

            all_embeddings = nx.get_node_attributes(G,'emb')
            x = np.array([val for (key,val) in all_embeddings.items()])
            x = torch.from_numpy(x).to(torch.float)
            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            data = Data(edge_index=edge_index, x=x)
            G_len_list.append(len(G))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if self.ego:
                ego_dir = self.processed_dir + '/ego/'
                if not os.path.exists(ego_dir):
                    os.makedirs(ego_dir)
                torch.save(data, ego_dir + 'ego_data_{}.pt'.format(i))
            else:
                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

        print("Avg %d, Min G %d, Max G %d", (sum(G_len_list)/len(G_len_list),\
                min(G_len_list), max(G_len_list)))

    def get(self, idx):
        if self.ego:
            data = torch.load(osp.join(self.processed_dir, 'ego/ego_data_{}.pt'.format(idx)))
        else:
            data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


