import torch
import math
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,PPI
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv, GAE, VGAE
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.distributions import Normal
from torch import nn
from .layers import MetaGCNConv, MetaGatedGraphConv, MetaGRUCell, MetaGatedGCNConv
import torch.nn.functional as F
from utils.utils import uniform
import ipdb

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class Encoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

class MetaMLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaMLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        if args.model in ['GAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
        elif args.model in ['VGAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
            self.fc_logvar = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(F.linear(x, weights['encoder.fc1.weight'],weights['encoder.fc1.bias']))
        if self.args.model in ['GAE']:
            return F.relu(F.linear(x, weights['encoder.fc_mu.weight'],weights['encoder.fc_mu.bias']))
        elif self.args.model in ['VGAE']:
            return F.relu(F.linear(x,weights['encoder.fc_mu.weight'],\
                    weights['encoder.fc_mu.bias'])),F.relu(F.linear(x,\
                    weights['encoder.fc_logvar.weight'],weights['encoder.fc_logvar.bias']))

class MLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        self.fc2 = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class GraphSignature(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(GraphSignature, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
            self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)
        else:
            self.gated_conv1 = MetaGatedGraphConv(in_channels, args.num_gated_layers)
            self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc2 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc3 = nn.Linear(in_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, keys):
        if self.args.use_gcn_sig:
            x = F.relu(self.conv1(x, edge_index, \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))
        else:
            x = F.relu(self.gated_conv1(x, edge_index, weights,keys))

        x = x.sum(0)
        x_gamma_1 = F.linear(x, weights['encoder.signature.fc1.weight'],\
                weights['encoder.signature.fc1.bias'])
        x_beta_1 = F.linear(x, weights['encoder.signature.fc2.weight'],\
                weights['encoder.signature.fc2.bias'])
        x_gamma_2 = F.linear(x, weights['encoder.signature.fc3.weight'],\
                weights['encoder.signature.fc3.bias'])
        x_beta_2 = F.linear(x, weights['encoder.signature.fc4.weight'],\
                weights['encoder.signature.fc4.bias'])
        return torch.tanh(x_gamma_1), torch.tanh(x_beta_1),\
                torch.tanh(x_gamma_2), torch.tanh(x_beta_2)

class MetaSignatureEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaSignatureEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = MetaGCNConv(
                2 * out_channels, out_channels, cached=False)
        # in_channels is the input feature dim
        self.signature = GraphSignature(args, in_channels, out_channels)

    def forward(self, x, edge_index, weights, inner_loop=True):
        keys = list(weights.keys())
        sig_keys = [key for key in keys if 'signature' in key]
        if inner_loop:
            with torch.no_grad():
                sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)
                self.cache_sig_out = [sig_gamma_1,sig_beta_1,sig_gamma_2,sig_beta_2]
        else:
            sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)
            self.cache_sig_out = [sig_gamma_1,sig_beta_1,sig_gamma_2,sig_beta_2]

        x = F.relu(self.conv1(x, edge_index, weights['encoder.conv1.weight'],\
                weights['encoder.conv1.bias'], gamma=sig_gamma_1, beta=sig_beta_1))
        if self.args.layer_norm:
            x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        if self.args.model in ['GAE']:
            x = self.conv2(x, edge_index,weights['encoder.conv2.weight'],\
                    weights['encoder.conv2.bias'],gamma=sig_gamma_2, beta=sig_beta_2)
            if self.args.layer_norm:
                x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
            return x
        elif self.args.model in ['VGAE']:
            mu = self.conv_mu(x,edge_index,weights['encoder.conv_mu.weight'],\
                    weights['encoder.conv_mu.bias'], gamma=sig_gamma_2, beta=sig_beta_2)
            sig = self.conv_logvar(x,edge_index,weights['encoder.conv_logvar.weight'],\
                weights['encoder.conv_logvar.bias'], gamma=sig_gamma_2, beta=sig_beta_2)
            if self.args.layer_norm:
                mu = nn.LayerNorm(mu.size()[1:], elementwise_affine=False)(mu)
                sig = nn.LayerNorm(sig.size()[1:], elementwise_affine=False)(sig)
            return mu, sig

class MetaGatedSignatureEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaGatedSignatureEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGatedGCNConv(in_channels, 2 * out_channels, gating=args.gating, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGatedGCNConv(2 * out_channels, out_channels, gating=args.gating, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGatedGCNConv(2 * out_channels, out_channels, gating=args.gating, cached=False)
            self.conv_logvar = MetaGatedGCNConv(
                2 * out_channels, out_channels, gating=args.gating, cached=False)
        # in_channels is the input feature dim
        self.signature = GraphSignature(args, in_channels, out_channels)

    def forward(self, x, edge_index, weights, inner_loop=True):
        keys = list(weights.keys())
        sig_keys = [key for key in keys if 'signature' in key]
        if inner_loop:
            with torch.no_grad():
                sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)
                self.cache_sig_out = [sig_gamma_1,sig_beta_1,sig_gamma_2,sig_beta_2,\
                                      torch.sigmoid(weights['encoder.conv1.gating_weights']),\
                                      torch.sigmoid(weights['encoder.conv_mu.gating_weights']),\
                                      torch.sigmoid(weights['encoder.conv_logvar.gating_weights'])]
        else:
            sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)

        x = F.relu(self.conv1(x, edge_index,\
                weights['encoder.conv1.weight_1'],\
                weights['encoder.conv1.weight_2'],\
                weights['encoder.conv1.bias'],\
                weights['encoder.conv1.gating_weights'],\
                gamma=sig_gamma_1, beta=sig_beta_1))
        if self.args.layer_norm:
            x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        if self.args.model in ['GAE']:
            x =  self.conv2(x, edge_index,\
                    weights['encoder.conv_mu.weight_1'],\
                    weights['encoder.conv_mu.weight_2'],\
                    weights['encoder.conv_mu.bias'],\
                    weights['encoder.conv_mu.gating_weights'],\
                    gamma=sig_gamma_2, beta=sig_beta_2)
            if self.args.layer_norm:
                x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
            return x
        elif self.args.model in ['VGAE']:
            mu = self.conv_mu(x,edge_index,\
                    weights['encoder.conv_mu.weight_1'],\
                    weights['encoder.conv_mu.weight_2'],\
                    weights['encoder.conv_mu.bias'],\
                    weights['encoder.conv_mu.gating_weights'],\
                    gamma=sig_gamma_2, beta=sig_beta_2)
            sig = self.conv_logvar(x,edge_index,\
                    weights['encoder.conv_logvar.weight_1'],\
                    weights['encoder.conv_logvar.weight_2'],\
                    weights['encoder.conv_logvar.bias'],\
                    weights['encoder.conv_logvar.gating_weights'],\
                    gamma=sig_gamma_2, beta=sig_beta_2)
            if self.args.layer_norm:
                mu = nn.LayerNorm(mu.size()[1:], elementwise_affine=False)(mu)
                sig = nn.LayerNorm(sig.size()[1:], elementwise_affine=False)(sig)
            return mu, sig

class MetaEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = MetaGCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index,\
                    weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x,edge_index,weights['encoder.conv_mu.weight'],\
                    weights['encoder.conv_mu.bias']),\
                self.conv_logvar(x,edge_index,weights['encoder.conv_logvar.weight'],\
                weights['encoder.conv_logvar.bias'])

class Net(torch.nn.Module):
    def __init__(self,train_dataset):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x

