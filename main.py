from comet_ml import Experiment
import torch
import torch.nn.functional as F
import os
import sys
import os.path as osp
import argparse
from data.data import load_dataset
from torch_geometric.datasets import Planetoid,PPI,TUDataset
import torch_geometric.transforms as T
import json
from torch_geometric.nn import GATConv, GCNConv
from models.autoencoder import MyGAE, MyVGAE
from torch_geometric.data import DataLoader
from maml import meta_gradient_step
from models.models import *
from utils.utils import global_test, test, EarlyStopping, seed_everything,\
        filter_state_dict, create_nx_graph, calc_adamic_adar_score,\
        create_nx_graph_deepwalk, train_deepwalk_model,calc_deepwalk_score
from utils.utils import run_analysis
from collections import OrderedDict
from torchviz import make_dot
import numpy as np
import wandb
import ipdb

def test(args,meta_model,optimizer,test_loader,train_epoch,return_val=False,inner_steps=10,seed= 0):
    ''' Meta-Testing '''
    mode='Test'
    test_graph_id_local = 0
    test_graph_id_global = 0
    args.resplit = False
    epoch=0
    args.final_test = False
    inner_test_auc_array = None
    inner_test_ap_array = None
    if return_val:
        args.inner_steps = inner_steps
        args.final_test = True
        inner_test_auc_array = np.zeros((len(test_loader)*args.test_batch_size, int(1000/5)))
        inner_test_ap_array = np.zeros((len(test_loader)*args.test_batch_size, int(1000/5)))

    meta_loss = torch.Tensor([0])
    test_avg_auc_list, test_avg_ap_list = [], []
    test_inner_avg_auc_list, test_inner_avg_ap_list = [], []
    for j,data in enumerate(test_loader):
        if args.adamic_adar_baseline:
            # Val Ratio is Fixed at 0.1
            meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
            data = meta_model.split_edges(data[0],val_ratio=args.meta_val_edge_ratio,\
                    test_ratio=meta_test_edge_ratio)
            G_test = create_nx_graph(data)
            auc, ap = calc_adamic_adar_score(G_test,data.test_pos_edge_index,data.test_neg_edge_index)
            test_avg_auc_list.append(auc)
            test_avg_ap_list.append(ap)
            test_graph_id_global += 1
            continue
        if args.deepwalk_baseline or args.deepwalk_and_mlp:
            # Val Ratio is Fixed at 0.2
            meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
            data = meta_model.split_edges(data[0], val_ratio=args.meta_val_edge_ratio, \
                                          test_ratio=meta_test_edge_ratio)
            G = create_nx_graph_deepwalk(data)
            node_vectors, entity2index, index2entity = train_deepwalk_model(G,seed=seed)
            if args.deepwalk_and_mlp:
                early_stopping = EarlyStopping(patience=args.patience, verbose=False)
                input_dim = args.num_features + node_vectors.shape[1]
                mlp = MLPEncoder(args, input_dim,
                                 args.num_channels).to(args.dev)
                mlp_optimizer = torch.optim.Adam(mlp.parameters(),
                                                 lr=args.mlp_lr)
                # node1 = data.x[torch.tensor(list(entity2index.keys())).long()]
                all_node_list = list(range(0, len(data.x)))
                node_order = [entity2index[node_i] for node_i in all_node_list]
                node1 = torch.tensor(node_vectors[node_order])
                for mlp_epochs in range(0, args.epochs):
                    mlp_optimizer.zero_grad()
                    # node_inp = torch.cat([torch.tensor(node_vectors), node1], dim=1)
                    node_inp = torch.cat([data.x, node1], dim=1)
                    node_inp = node_inp.to(args.dev)
                    z = mlp(node_inp, edge_index=None)
                    loss = meta_model.recon_loss(z, data.train_pos_edge_index.cuda())
                    loss.backward()
                    mlp_optimizer.step()
                    if mlp_epochs % 10 == 0:
                        if mlp_epochs % 50 == 0:
                            print("Epoch %d, Loss: %f" %(mlp_epochs, loss))
                        with torch.no_grad():
                            val_auc, val_ap = meta_model.test(z, data.val_pos_edge_index,
                                                 data.val_neg_edge_index)
                        early_stopping(val_auc, meta_model)
                    if early_stopping.early_stop:
                        print("Early stopping for Graph %d | AUC: %f AP: %f" \
                                %(test_graph_id_global, val_auc, val_ap))
                        break

                node_inp = torch.cat([data.x, node1], dim=1)
                # node_inp = torch.cat([torch.tensor(node_vectors), node1], dim=1)
                node_inp = node_inp.to(args.dev)
                node_vectors = mlp(node_inp, edge_index=None)
                auc, ap = meta_model.test(z, data.test_pos_edge_index,
                                     data.test_neg_edge_index)
            else:
                node_vectors = node_vectors.detach().cpu().numpy()
                auc, ap = calc_deepwalk_score(data.test_pos_edge_index,
                                              data.test_neg_edge_index,
                                              node_vectors,entity2index)

            print("Graph %d| Test AUC: %f AP: %f" %(test_graph_id_global, auc, ap))
            test_avg_auc_list.append(auc)
            test_avg_ap_list.append(ap)
            test_graph_id_global += 1
            continue

        if not args.random_baseline and not args.adamic_adar_baseline:
            test_graph_id_local, meta_loss, test_inner_avg_auc_list, test_inner_avg_ap_list = meta_gradient_step(meta_model,\
                    args,data,optimizer,args.inner_steps,args.inner_lr,args.order,test_graph_id_local,mode,\
                    test_inner_avg_auc_list, test_inner_avg_ap_list,epoch,j,False,\
                            inner_test_auc_array,inner_test_ap_array)
        auc_list, ap_list = global_test(args,meta_model,data,OrderedDict(meta_model.named_parameters()))
        test_avg_auc_list.append(sum(auc_list)/len(auc_list))
        test_avg_ap_list.append(sum(ap_list)/len(ap_list))

        ''' Test Logging '''
        if args.comet:
            if len(auc_list) > 0 and len(ap_list) > 0:
                auc_metric = 'Test_Outer_Batch_Graph_' + str(j) +'_AUC'
                ap_metric = 'Test_Outer_Batch_Graph_' + str(j) +'_AP'
                args.experiment.log_metric(auc_metric,sum(auc_list)/len(auc_list),step=train_epoch)
                args.experiment.log_metric(ap_metric,sum(ap_list)/len(ap_list),step=train_epoch)
        if args.wandb:
            if len(auc_list) > 0 and len(ap_list) > 0:
                auc_metric = 'Test_Outer_Batch_Graph_' + str(j) +'_AUC'
                ap_metric = 'Test_Outer_Batch_Graph_' + str(j) +'_AP'
                wandb.log({auc_metric:sum(auc_list)/len(auc_list),\
                        ap_metric:sum(ap_list)/len(ap_list),"x":epoch},commit=False)

    print("Failed on %d graphs" %(args.fail_counter))
    print("Epoch: %d | Test Global Avg Auc %f | Test Global Avg AP %f" \
            %(train_epoch, sum(test_avg_auc_list)/len(test_avg_auc_list),\
                    sum(test_avg_ap_list)/len(test_avg_ap_list)))
    if args.comet:
        if len(auc_list) > 0 and len(ap_list) > 0:
            auc_metric = 'Test_Avg_' +'_AUC'
            ap_metric = 'Test_Avg_' +'_AP'
            inner_auc_metric = 'Test_Inner_Avg' +'_AUC'
            inner_ap_metric = 'Test_Inner_Avg' +'_AP'
            args.experiment.log_metric(auc_metric,sum(test_avg_auc_list)/len(test_avg_auc_list),step=train_epoch)
            args.experiment.log_metric(ap_metric,sum(test_avg_ap_list)/len(test_avg_ap_list),step=train_epoch)
            args.experiment.log_metric(inner_auc_metric,sum(test_inner_avg_auc_list)/len(test_inner_avg_auc_list),step=train_epoch)
            args.experiment.log_metric(inner_ap_metric,sum(test_inner_avg_ap_list)/len(test_inner_avg_ap_list),step=train_epoch)
    if args.wandb:
        if len(test_avg_auc_list) > 0 and len(test_avg_ap_list) > 0:
            auc_metric = 'Test_Avg' +'_AUC'
            ap_metric = 'Test_Avg' +'_AP'
            wandb.log({auc_metric:sum(test_avg_auc_list)/len(test_avg_auc_list),\
                    ap_metric:sum(test_avg_ap_list)/len(test_avg_ap_list),\
                    "x":train_epoch},commit=False)
        if len(test_inner_avg_auc_list) > 0 and len(test_inner_avg_ap_list) > 0:
            inner_auc_metric = 'Test_Inner_Avg' +'_AUC'
            inner_ap_metric = 'Test_Inner_Avg' +'_AP'
            wandb.log({inner_auc_metric:sum(test_inner_avg_auc_list)/len(test_inner_avg_auc_list),
                    inner_ap_metric:sum(test_inner_avg_ap_list)/len(test_inner_avg_ap_list),
                    "x":train_epoch},commit=False)
    if len(test_inner_avg_ap_list) > 0:
        print('Epoch {:01d} | Test Inner AUC: {:.4f}, AP: {:.4f}'.format(train_epoch,sum(test_inner_avg_auc_list)/len(test_inner_avg_auc_list),sum(test_inner_avg_ap_list)/len(test_inner_avg_ap_list)))

    if return_val:
        test_avg_auc = sum(test_avg_auc_list)/len(test_avg_auc_list)
        test_avg_ap = sum(test_avg_ap_list)/len(test_avg_ap_list)
        if len(test_inner_avg_ap_list) > 0:
            test_inner_avg_auc = sum(test_inner_avg_auc_list)/len(test_inner_avg_auc_list)
            test_inner_avg_ap = sum(test_inner_avg_ap_list)/len(test_inner_avg_ap_list)
        #Remove All zero rows
        test_auc_array = inner_test_auc_array[~np.all(inner_test_auc_array == 0, axis=1)]
        test_ap_array = inner_test_ap_array[~np.all(inner_test_ap_array == 0, axis=1)]
        test_aggr_auc = np.sum(test_auc_array,axis=0)/len(test_loader)
        test_aggr_ap = np.sum(test_ap_array,axis=0)/len(test_loader)
        max_auc = np.max(test_aggr_auc)
        max_ap = np.max(test_aggr_ap)
        auc_metric = 'Test_Complete' +'_AUC'
        ap_metric = 'Test_Complete' +'_AP'
        for val_idx in range(0,test_auc_array.shape[1]):
            auc = test_aggr_auc[val_idx]
            ap = test_aggr_ap[val_idx]
            if args.comet:
                args.experiment.log_metric(auc_metric,auc,step=val_idx)
                args.experiment.log_metric(ap_metric,ap,step=val_idx)
            if args.wandb:
                wandb.log({auc_metric:auc,ap_metric:ap,"x":val_idx})
        print("Test Max AUC :%f | Test Max AP: %f" %(max_auc,max_ap))

        ''' Save Local final params '''
        if not os.path.exists('../saved_models/'):
            os.makedirs('../saved_models/')
        save_path = '../saved_models/' + args.namestr + '_local.pt'
        torch.save(meta_model.state_dict(), save_path)
        return max_auc, max_ap

def validation(args,meta_model,optimizer,val_loader,train_epoch,return_val=False):
    ''' Meta-Valing '''
    mode='Val'
    val_graph_id_local = 0
    val_graph_id_global = 0
    args.resplit = True
    epoch=0
    meta_loss = torch.Tensor([0])
    val_avg_auc_list, val_avg_ap_list = [], []
    val_inner_avg_auc_list, val_inner_avg_ap_list = [], []
    args.final_val = False
    inner_val_auc_array = None
    inner_val_ap_array = None
    if return_val:
        args.inner_steps = inner_steps
        args.final_val = True
        inner_val_auc_array = np.zeros((len(val_loader)*args.val_batch_size, int(1000/5)))
        inner_val_ap_array = np.zeros((len(val_loader)*args.val_batch_size, int(1000/5)))
    for j,data in enumerate(val_loader):
        if not args.random_baseline:
            val_graph_id_local, meta_loss, val_inner_avg_auc_list, val_inner_avg_ap_list = meta_gradient_step(meta_model,\
                    args,data,optimizer,args.inner_steps,args.inner_lr,args.order,val_graph_id_local,mode,\
                    val_inner_avg_auc_list,val_inner_avg_ap_list,epoch,j,False,\
                            inner_val_auc_array,inner_val_ap_array)
        auc_list, ap_list = global_test(args,meta_model,data,OrderedDict(meta_model.named_parameters()))
        val_avg_auc_list.append(sum(auc_list)/len(auc_list))
        val_avg_ap_list.append(sum(ap_list)/len(ap_list))
        if args.comet:
            if len(auc_list) > 0 and len(ap_list) > 0:
                auc_metric = 'Val_Batch_Graph_' + str(j) +'_AUC'
                ap_metric = 'Val_Batch_Graph_' + str(j) +'_AP'
                args.experiment.log_metric(auc_metric,sum(auc_list)/len(auc_list),step=train_epoch)
                args.experiment.log_metric(ap_metric,sum(ap_list)/len(ap_list),step=train_epoch)
        if args.wandb:
            if len(auc_list) > 0 and len(ap_list) > 0:
                auc_metric = 'Val_Batch_Graph_' + str(j) +'_AUC'
                ap_metric = 'Val_Batch_Graph_' + str(j) +'_AP'
                wandb.log({auc_metric:sum(auc_list)/len(auc_list),\
                        ap_metric:sum(ap_list)/len(ap_list),"x":epoch},commit=False)

    print("Val Avg Auc %f | Val Avg AP %f" %(sum(val_avg_auc_list)/len(val_avg_auc_list),\
            sum(val_avg_ap_list)/len(val_avg_ap_list)))
    if len(val_inner_avg_ap_list) > 0:
        print("Val Inner Avg Auc %f | Val Avg AP %f" %(sum(val_inner_avg_auc_list)/len(val_inner_avg_auc_list),\
                sum(val_inner_avg_ap_list)/len(val_inner_avg_ap_list)))
    if args.comet:
        if len(auc_list) > 0 and len(ap_list) > 0:
            auc_metric = 'Val_Avg_' +'_AUC'
            ap_metric = 'Val_Avg_' +'_AP'
            inner_auc_metric = 'Val_Inner_Avg' +'_AUC'
            inner_ap_metric = 'Val_Inner_Avg' +'_AP'
            args.experiment.log_metric(auc_metric,sum(val_avg_auc_list)/len(val_avg_auc_list),step=train_epoch)
            args.experiment.log_metric(ap_metric,sum(val_avg_ap_list)/len(val_avg_ap_list),step=train_epoch)
            args.experiment.log_metric(inner_auc_metric,sum(val_inner_avg_auc_list)/len(val_inner_avg_auc_list),step=train_epoch)
            args.experiment.log_metric(inner_ap_metric,sum(val_inner_avg_ap_list)/len(val_inner_avg_ap_list),step=train_epoch)
    if args.wandb:
        if len(auc_list) > 0 and len(ap_list) > 0:
            auc_metric = 'Val_Avg' +'_AUC'
            ap_metric = 'Val_Avg' +'_AP'
            inner_auc_metric = 'Val_Inner_Avg' +'_AUC'
            inner_ap_metric = 'Val_Inner_Avg' +'_AP'
            wandb.log({auc_metric:sum(val_avg_auc_list)/len(val_avg_auc_list),\
                    ap_metric:sum(val_avg_ap_list)/len(val_avg_ap_list),\
                    inner_auc_metric:sum(val_inner_avg_auc_list)/len(val_inner_avg_auc_list),\
                    inner_ap_metric:sum(val_inner_avg_ap_list)/len(val_inner_avg_ap_list),\
                    "x":epoch},commit=False)

    if return_val:
        val_avg_auc = sum(val_avg_auc_list)/len(val_avg_auc_list)
        val_avg_ap = sum(val_avg_ap_list)/len(val_avg_ap_list)
        val_inner_avg_auc = sum(val_inner_avg_auc_list)/len(val_inner_avg_auc_list)
        val_inner_avg_ap = sum(val_inner_avg_ap_list)/len(val_inner_avg_ap_list)
        #Remove All zero rows
        val_auc_array = inner_val_auc_array[~np.all(inner_val_auc_array == 0, axis=1)]
        val_ap_array = inner_val_ap_array[~np.all(inner_val_ap_array == 0, axis=1)]

        val_aggr_auc = np.sum(val_auc_array,axis=0)/len(val_loader)
        val_aggr_ap = np.sum(val_ap_array,axis=0)/len(val_loader)
        max_auc = np.max(val_aggr_auc)
        max_ap = np.max(val_aggr_ap)
        auc_metric = 'Val_Complete' +'_AUC'
        ap_metric = 'Val_Complete' +'_AP'
        for val_idx in range(0,val_auc_array.shape[1]):
            auc = val_aggr_auc[val_idx]
            ap = val_aggr_ap[val_idx]
            if args.comet:
                args.experiment.log_metric(auc_metric,auc,step=val_idx)
                args.experiment.log_metric(ap_metric,ap,step=val_idx)
            if args.wandb:
                wandb.log({auc_metric:auc,ap_metric:ap,"x":val_idx})
        print("Val Max AUC :%f | Val Max AP: %f" %(max_auc,max_ap))
        return max_auc, max_ap

def main(args):
    assert args.model in ['GAE', 'VGAE']
    kwargs = {'GAE': MyGAE, 'VGAE': MyVGAE}
    kwargs_enc = {'GCN': MetaEncoder, 'FC': MLPEncoder, 'MLP': MetaMLPEncoder,
                  'GraphSignature': MetaSignatureEncoder,
                  'GatedGraphSignature': MetaGatedSignatureEncoder}

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
    train_loader, val_loader, test_loader = load_dataset(args.dataset,args)
    meta_model = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
    if args.train_only_gs:
        trainable_parameters = []
        for name, p in meta_model.named_parameters():
            if "signature" in name:
                trainable_parameters.append(p)
            else:
                p.requires_grad = False
        optimizer = torch.optim.Adam(trainable_parameters, lr=args.meta_lr)
    else:
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)

    total_loss = 0
    if not args.do_kl_anneal:
        args.kl_anneal = 1

    if args.encoder == 'GraphSignature' or args.encoder == 'GatedGraphSignature':
        args.allow_unused = True
    else:
        args.allow_unused = False

    ''' Random or Adamic Adar Baseline '''
    if args.random_baseline or args.adamic_adar_baseline or args.deepwalk_baseline:
        test_inner_avg_auc, test_inner_avg_ap = test(args,meta_model,optimizer,test_loader,0,\
                return_val=True,inner_steps=1000,seed=args.seed)
        sys.exit()

    ''' Run WL-Kernel '''
    if args.wl:
        load_path = '../saved_models/' + args.namestr + '.pt'
        meta_model.load_state_dict(torch.load(load_path))
        run_analysis(args, meta_model,train_loader)
        test(args,meta_model,optimizer,test_loader,0)
        sys.exit()

    ''' Meta-training '''
    mode = 'Train'
    meta_loss = torch.Tensor([0])
    args.final_test = False
    for epoch in range(0,args.epochs):
        graph_id_local = 0
        graph_id_global = 0
        train_inner_avg_auc_list, train_inner_avg_ap_list = [], []
        if epoch > 0 and args.dataset !='PPI':
            args.resplit = False
        for i,data in enumerate(train_loader):
            if args.debug:
                ''' Print the Computation Graph '''
                dot = make_dot(meta_gradient_step(meta_model,args,data,optimizer,args.inner_steps,args.inner_lr,\
                        args.order,graph_id_local,mode,test_inner_avg_auc_list, test_inner_avg_ap_list, \
                        epoch,i,True)[1],params=dict(meta_model.named_parameters()))
                dot.format = 'png'
                dot.render(args.debug_name)
                quit()

            graph_id_local, meta_loss, train_inner_avg_auc_list, train_inner_avg_ap_list = meta_gradient_step(meta_model,\
                    args,data,optimizer,args.inner_steps,args.inner_lr,args.order,graph_id_local,\
                    mode,train_inner_avg_auc_list, train_inner_avg_ap_list,epoch,i,True)
            if args.do_kl_anneal:
                args.kl_anneal = args.kl_anneal + 1/args.epochs

            auc_list, ap_list = global_test(args,meta_model,data,OrderedDict(meta_model.named_parameters()))
            if args.comet:
                if len(ap_list) > 0:
                    auc_metric = 'Train_Global_Batch_Graph_' + str(i) +'_AUC'
                    ap_metric = 'Train_Global_Batch_Graph_' + str(i) +'_AP'
                    args.experiment.log_metric(auc_metric,sum(auc_list)/len(auc_list),step=epoch)
                    args.experiment.log_metric(ap_metric,sum(ap_list)/len(ap_list),step=epoch)
            if args.wandb:
                if len(ap_list) > 0:
                        auc_metric = 'Train_Global_Batch_Graph_' + str(i) +'_AUC'
                        ap_metric = 'Train_Global_Batch_Graph_' + str(i) +'_AP'
                        wandb.log({auc_metric:sum(auc_list)/len(auc_list),\
                                ap_metric:sum(ap_list)/len(ap_list),"x":epoch},commit=False)
            graph_id_global += len(ap_list)

            if args.wandb:
                wandb.log()

        if args.comet:
            if len(train_inner_avg_ap_list) > 0:
                auc_metric = 'Train_Inner_Avg' +'_AUC'
                ap_metric = 'Train_Inner_Avg' + str(i) +'_AP'
                args.experiment.log_metric(auc_metric,sum(train_inner_avg_auc_list)/len(train_inner_avg_auc_list),step=epoch)
                args.experiment.log_metric(ap_metric,sum(train_inner_avg_ap_list)/len(train_inner_avg_ap_list),step=epoch)
        if args.wandb:
            if len(train_inner_avg_ap_list) > 0:
                    auc_metric = 'Train_Inner_Avg' +'_AUC'
                    ap_metric = 'Train_Inner_Avg' + str(i) +'_AP'
                    wandb.log({auc_metric:sum(train_inner_avg_auc_list)/len(train_inner_avg_auc_list),\
                            ap_metric:sum(train_inner_avg_ap_list)/len(train_inner_avg_ap_list),\
                            "x":epoch},commit=False)

        if len(train_inner_avg_ap_list) > 0:
            print('Train Inner AUC: {:.4f}, AP: {:.4f}'.format(sum(train_inner_avg_auc_list)/len(train_inner_avg_auc_list),\
                            sum(train_inner_avg_ap_list)/len(train_inner_avg_ap_list)))

        ''' Meta-Testing After every Epoch'''
        meta_model_copy = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
        meta_model_copy.load_state_dict(meta_model.state_dict())
        if args.train_only_gs:
            optimizer_copy = torch.optim.Adam(trainable_parameters, lr=args.meta_lr)
        else:
            optimizer_copy = torch.optim.Adam(meta_model_copy.parameters(), lr=args.meta_lr)
        optimizer_copy.load_state_dict(optimizer.state_dict())
        validation(args,meta_model_copy,optimizer_copy,val_loader,epoch)
        test(args,meta_model_copy,optimizer_copy,test_loader,epoch,inner_steps=args.inner_steps)

    print("Failed on %d Training graphs" %(args.fail_counter))

    ''' Save Global Params '''
    if not os.path.exists('../saved_models/'):
        os.makedirs('../saved_models/')
    save_path = '../saved_models/meta_vgae.pt'
    save_path = '../saved_models/' + args.namestr + '_global_.pt'
    torch.save(meta_model.state_dict(), save_path)

    ''' Run to Convergence '''
    if args.ego:
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
        args.inner_lr = args.inner_lr * args.reset_inner_factor
    val_inner_avg_auc, val_inner_avg_ap = test(args,meta_model,optimizer,val_loader,epoch,\
            return_val=True,inner_steps=1000)
    if args.ego:
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
        args.inner_lr = args.inner_lr * args.reset_inner_factor
    test_inner_avg_auc, test_inner_avg_ap = test(args,meta_model,optimizer,test_loader,epoch,\
            return_val=True,inner_steps=1000)
    if args.comet:
        args.experiment.end()

    val_eval_metric = 0.5*val_inner_avg_auc + 0.5*val_inner_avg_ap
    test_eval_metric = 0.5*test_inner_avg_auc + 0.5*test_inner_avg_ap
    return val_eval_metric

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VGAE')
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--num_channels', type=int, default='16')
    parser.add_argument('--dataset', type=str, default='PPI')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--num_fixed_features', default=20, type=int)
    parser.add_argument('--num_concat_features', default=10, type=int)
    parser.add_argument('--meta_train_edge_ratio', type=float, default='0.2')
    parser.add_argument('--meta_val_edge_ratio', type=float, default='0.2')
    parser.add_argument('--k_core', type=int, default=5, help="K-core for Graph")
    parser.add_argument('--clip', type=float, default='1', help='Gradient Clip')
    parser.add_argument('--clip_weight_val', type=float, default='0.1',\
            help='Weight Clip')
    parser.add_argument('--train_ratio', type=float, default='0.8', \
            help='Used to split number of graphs for training if not provided')
    parser.add_argument('--val_ratio', type=float, default='0.1',\
            help='Used to split number of graphs for va1idation if not provided')
    parser.add_argument('--num_gated_layers', default=4, type=int,\
            help='Number of layers to use for the Gated Graph Conv Layer')
    parser.add_argument('--mlp_lr', default=1e-3, type=float)
    parser.add_argument('--inner-lr', default=0.01, type=float)
    parser.add_argument('--reset_inner_factor', default=20, type=float)
    parser.add_argument('--meta-lr', default=0.001, type=float)
    parser.add_argument('--order', default=2, type=int, help='MAML gradient order')
    parser.add_argument('--inner_steps', type=int, default=50)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--concat_fixed_feats", action="store_true", default=False,
		help='Concatenate random node features to current node features')
    parser.add_argument("--extra_backward", action="store_true", default=False,
		help='Do Extra Backward pass like in Original Pytorch MAML repo')
    parser.add_argument("--use_fixed_feats", action="store_true", default=False,
		help='Use a random node features')
    parser.add_argument("--use_same_fixed_feats", action="store_true", default=False,
		help='Use a random node features for all nodes')
    parser.add_argument('--min_nodes', type=int, default=1000, \
            help='Min Nodes needed for a graph to be included')
    parser.add_argument('--max_nodes', type=int, default=50000, \
            help='Max Nodes needed for a graph to be included')
    parser.add_argument('--kl_anneal', type=int, default=0, \
            help='KL Anneal Coefficient')
    parser.add_argument('--do_kl_anneal', action="store_true", default=False, \
            help='Do KL Annealing')
    parser.add_argument("--clip_grad", action="store_true", default=False,
		help='Gradient Clipping')
    parser.add_argument("--clip_weight", action="store_true", default=False,
		help='Weight Clipping')
    parser.add_argument("--use_gcn_sig", action="store_true", default=False,
		help='Use GCN in Signature Function')
    parser.add_argument("--train_only_gs", action="store_true", default=False,
		help='Train only the Graph Signature Function')
    parser.add_argument("--random_baseline", action="store_true", default=False,
		help='Use a Random Baseline')
    parser.add_argument("--adamic_adar_baseline", action="store_true", default=False,
		help='Use Adamic-Adar Baseline')
    parser.add_argument("--deepwalk_baseline", action ="store_true", default=False,
        help = "Use Deepwalk Baseline")
    parser.add_argument("--deepwalk_and_mlp", action ="store_true", default=False,
        help = "Use Deepwalk Baseline and ML and MLPP")
    parser.add_argument("--reprocess", action="store_true", default=False,
		help='Reprocess AMINER dataset')
    parser.add_argument("--ego", action="store_true", default=False,
		help='Reprocess AMINER dataset as ego')
    parser.add_argument("--wl", action="store_true", default=False,
		help='Run WL-Kernel on dataset')
    parser.add_argument("--comet", action="store_true", default=False,
		help='Use comet for logging')
    parser.add_argument("--wandb", action="store_true", default=False,
		help='Use wandb for logging')
    parser.add_argument("--comet_username", type=str, default="joeybose",
                help='Username for comet logging')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--comet_apikey", type=str,\
            help='Api for comet logging')
    parser.add_argument('--patience', type=int, default=40, help="Early Stopping")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug')
    parser.add_argument('--opus', default=False, action='store_true',
                        help='Change AMINER File Path for Opus')
    parser.add_argument('--debug_name', type=str, default="one_maml_graph",
                        help='where to save/load')
    parser.add_argument('--namestr', type=str, default='Meta-Graph', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--study_uid', type=str, default='')
    parser.add_argument('--gating', type=str, default=None, choices=[None, 'signature', 'weights', 'signature_cond', 'weights_cond'])
    parser.add_argument('--layer_norm', default=False, action='store_true',
                        help='use layer norm')
    args = parser.parse_args()

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.comet_apikey = data["apikey"]
        args.comet_username = data["username"]
        args.wandb_apikey = data["wandbapikey"]

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset=='PPI':
        project_name = 'meta-graph-ppi'
    elif args.dataset=='REDDIT-MULTI-12K':
        project_name = "meta-graph-reddit"
    elif args.dataset=='FIRSTMM_DB':
        project_name = "meta-graph-firstmmdb"
    elif args.dataset=='DD':
        project_name = "meta-graph-dd"
    elif args.dataset=='AMINER':
        project_name = "meta-graph-aminer"
    else:
        project_name='meta-graph'

    if args.comet:
        experiment = Experiment(api_key=args.comet_apikey,\
                project_name=project_name,\
                workspace=args.comet_username)
        experiment.set_name(args.namestr)
        args.experiment = experiment

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project=project_name,name=args.namestr)

    print(vars(args))
    eval_metric = main(args)
