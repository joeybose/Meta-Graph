import torch
from torch import autograd
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union
from utils.utils import custom_clip_grad_norm_, val, test, EarlyStopping, monitor_grad_norm, monitor_grad_norm_2, monitor_weight_norm
from torchviz import make_dot
import ipdb
import wandb
from copy import deepcopy

def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model,
                       args,
                       data_batch,
                       optimiser,
                       inner_train_steps,
                       inner_lr,
                       order,
                       graph_id,
                       mode,
                       inner_avg_auc_list,
                       inner_avg_ap_list,
                       epoch,
                       batch_id,
                       train,
                       inner_test_auc_array=None,
                       inner_test_ap_array=None):
    """
    Perform a gradient step on a meta-learner.
    # Arguments
        model: Base model of the meta-learner being trained
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        data_batch: Input samples for all few shot tasks
            meta-gradients after applying the update to
        inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
        inner_lr: Learning rate used to update the fast weights on the inner update
        order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
            query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
            weights on the query with respect to the original weights).
        graph_id: The ID of graph currently being trained
        train: Whether to update the meta-learner weights at the end of the episode.
        inner_test_auc_array: Final Test AUC array where we train to convergence
        inner_test_ap_array: Final Test AP array where we train to convergence
    """

    task_gradients = []
    task_losses = []
    task_predictions = []
    auc_list = []
    ap_list = []
    torch.autograd.set_detect_anomaly(True)
    for idx, data_graph in enumerate(data_batch):
        data_graph.train_mask = data_graph.val_mask = data_graph.test_mask = data_graph.y = None
        data_graph.batch = None
        num_nodes = data_graph.num_nodes

        if args.use_fixed_feats:
            perm = torch.randperm(args.feats.size(0))
            perm_idx = perm[:num_nodes]
            data_graph.x = args.feats[perm_idx]
        elif args.use_same_fixed_feats:
            node_feats = args.feats[0].unsqueeze(0).repeat(num_nodes,1)
            data_graph.x = node_feats

        if args.concat_fixed_feats:
            if data_graph.x.shape[1] < args.num_features:
                concat_feats = torch.randn(num_nodes,args.num_concat_features,requires_grad=False)
                data_graph.x = torch.cat((data_graph.x,concat_feats),1)

        # Val Ratio is Fixed at 0.1
        meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio

        ''' Check if Data is split'''
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev), data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data_graph.x.cuda()
            data = model.split_edges(data_graph,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)

        # Additional Failure Checks for small graphs
        if data.val_pos_edge_index.size()[1] == 0 or data.test_pos_edge_index.size()[1] == 0:
            args.fail_counter += 1
            print("Failed on Graph %d" %(graph_id))
            continue

        try:
            x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
            test_pos_edge_index, test_neg_edge_index = data.test_pos_edge_index.to(args.dev),\
                    data.test_neg_edge_index.to(args.dev)
        except:
            print("Failed Splitting data on Graph %d" %(graph_id))
            continue

        data_shape = x.shape[2:]
        create_graph = (True if order == 2 else False) and train

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())
        early_stopping = EarlyStopping(patience=args.patience, verbose=False)

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # Perform update of model weights
            z = model.encode(x, train_pos_edge_index, fast_weights, inner_loop=True)
            loss = model.recon_loss(z, train_pos_edge_index)
            if args.model in ['VGAE']:
                kl_loss = args.kl_anneal*(1 / num_nodes) * model.kl_loss()
                loss = loss + kl_loss
                # print("Inner KL Loss: %f" %(kl_loss.item()))
            if not args.train_only_gs:
                gradients = torch.autograd.grad(loss, fast_weights.values(),\
                        allow_unused=args.allow_unused, create_graph=create_graph)
                gradients = [0 if grad is None else grad  for grad in gradients]
                if args.wandb:
                    wandb.log({"Inner_Train_loss":loss.item()})

                if args.clip_grad:
                    # for grad in gradients:
                    custom_clip_grad_norm_(gradients,args.clip)
                    grad_norm = monitor_grad_norm_2(gradients)
                    if args.wandb:
                        inner_grad_norm_metric = 'Inner_Grad_Norm'
                        wandb.log({inner_grad_norm_metric:grad_norm})

            ''' Only do this if its the final test set eval '''
            if args.final_test and inner_batch % 5 ==0:
                inner_test_auc, inner_test_ap = test(model, x, train_pos_edge_index,
                        data.test_pos_edge_index, data.test_neg_edge_index,fast_weights)
                val_pos_edge_index = data.val_pos_edge_index.to(args.dev)
                val_loss = val(model,args, x,val_pos_edge_index,data.num_nodes,fast_weights)
                early_stopping(val_loss, model)
                my_step = int(inner_batch / 5)
                inner_test_auc_array[graph_id][my_step] = inner_test_auc
                inner_test_ap_array[graph_id][my_step] = inner_test_ap

            # Update weights manually
            if not args.train_only_gs and args.clip_weight:
                fast_weights = OrderedDict(
                    (name, torch.clamp((param - inner_lr * grad),-args.clip_weight_val,args.clip_weight_val))
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
            elif not args.train_only_gs:
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )

            if early_stopping.early_stop:
                print("Early stopping for Graph %d | AUC: %f AP: %f" \
                        %(graph_id, inner_test_auc, inner_test_ap))
                my_step = int(epoch / 5)
                inner_test_auc_array[graph_id][my_step:,] = inner_test_auc
                inner_test_ap_array[graph_id][my_step:,] = inner_test_ap
                break

        # Do a pass of the model on the validation data from the current task
        val_pos_edge_index = data.val_pos_edge_index.to(args.dev)
        z_val = model.encode(x, val_pos_edge_index, fast_weights, inner_loop=False)
        loss_val = model.recon_loss(z_val, val_pos_edge_index)
        if args.model in ['VGAE']:
            kl_loss = args.kl_anneal*(1 / num_nodes) * model.kl_loss()
            # print("Outer KL Loss: %f" %(kl_loss.item()))
            loss_val = loss_val + kl_loss

        if args.wandb:
            wandb.log({"Inner_Val_loss":loss_val.item()})
            # print("Inner Val Loss %f" % (loss_val.item()))

        ##TODO: Is this backward call needed here? Not sure because original repo has it
        # https://github.com/oscarknagg/few-shot/blob/master/few_shot/maml.py#L84
        if args.extra_backward:
            loss_val.backward(retain_graph=True)

        # Get post-update accuracies
        auc, ap = test(model, x, train_pos_edge_index,
                data.test_pos_edge_index, data.test_neg_edge_index,fast_weights)

        auc_list.append(auc)
        ap_list.append(ap)
        inner_avg_auc_list.append(auc)
        inner_avg_ap_list.append(ap)

        # Accumulate losses and gradients
        graph_id += 1
        task_losses.append(loss_val)
        if order == 1:
            gradients = torch.autograd.grad(loss_val, fast_weights.values(), create_graph=create_graph)
            named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
            task_gradients.append(named_grads)

    if len(auc_list) > 0 and len(ap_list) > 0 and batch_id % 5 == 0:
        print('Epoch {:01d} Inner Graph Batch: {:01d}, Inner-Update AUC: {:.4f}, AP: {:.4f}'.format(epoch,batch_id,sum(auc_list)/len(auc_list),sum(ap_list)/len(ap_list)))
    if args.comet:
        if len(ap_list) > 0:
            auc_metric = mode + '_Local_Batch_Graph_' + str(batch_id) + '_AUC'
            ap_metric = mode + '_Local_Batch_Graph_' + str(batch_id) + '_AP'
            avg_auc_metric = mode + '_Inner_Batch_Graph' + '_AUC'
            avg_ap_metric = mode + '_Inner_Batch_Graph' + '_AP'
            args.experiment.log_metric(auc_metric,sum(auc_list)/len(auc_list),step=epoch)
            args.experiment.log_metric(ap_metric,sum(ap_list)/len(ap_list),step=epoch)
            args.experiment.log_metric(avg_auc_metric,sum(auc_list)/len(auc_list),step=epoch)
            args.experiment.log_metric(avg_ap_metric,sum(ap_list)/len(ap_list),step=epoch)
    if args.wandb:
        if len(ap_list) > 0:
            auc_metric = mode + '_Local_Batch_Graph_' + str(batch_id) + '_AUC'
            ap_metric = mode + '_Local_Batch_Graph_' + str(batch_id) + '_AP'
            avg_auc_metric = mode + '_Inner_Batch_Graph' + '_AUC'
            avg_ap_metric = mode + '_Inner_Batch_Graph' + '_AP'
            wandb.log({auc_metric:sum(auc_list)/len(auc_list),ap_metric:sum(ap_list)/len(ap_list),\
                    avg_auc_metric:sum(auc_list)/len(auc_list),avg_ap_metric:sum(ap_list)/len(ap_list)})

    meta_batch_loss = torch.Tensor([0])
    if order == 1:
        if train and len(task_losses) != 0:
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()}
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(sum_task_gradients, name))
                )

            model.train()
            optimiser.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            ## TODO: Double check if you really need functional forward here
            z_dummy = model.encode(torch.zeros(x.shape[0],x.shape[1]).float().cuda(), \
                    torch.zeros(train_pos_edge_index.shape[0],train_pos_edge_index.shape[1]).long().cuda(), fast_weights)
            loss = model.recon_loss(z_dummy,torch.zeros(train_pos_edge_index.shape[0],\
                    train_pos_edge_index.shape[1]).long().cuda())
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()
            meta_batch_loss = torch.stack(task_losses).mean()
        return graph_id, meta_batch_loss, inner_avg_auc_list, inner_avg_ap_list

    elif order == 2:
        if len(task_losses) != 0:
            model.train()
            optimiser.zero_grad()
            meta_batch_loss = torch.stack(task_losses).mean()

            if train:
                meta_batch_loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
                    grad_norm = monitor_grad_norm(model)
                    if args.wandb:
                        outer_grad_norm_metric = 'Outer_Grad_Norm'
                        wandb.log({outer_grad_norm_metric:grad_norm})

                optimiser.step()
                if args.clip_weight:
                    for p in model.parameters():
                        p.data.clamp_(-args.clip_weight_val,args.clip_weight_val)
        return graph_id, meta_batch_loss, inner_avg_auc_list, inner_avg_ap_list
    else:
        raise ValueError('Order must be either 1 or 2.')
