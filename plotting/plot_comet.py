from comet_ml import API
import argparse
import csv
import json
import os
from statistics import mean
import wandb
import matplotlib
import numpy as np
import ipdb

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid')
sns.set_palette('colorblind')
plt.rcParams['text.usetex'] = False

def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)

def ApplyFont(ax):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()

    text_size = 14.0

    for t in ticks:
        t.set_fontname('Times New Roman')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabe(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

SetPlotRC()

def connect_to_comet(comet_apikey,comet_restapikey,comet_username,comet_project):
    if os.path.isfile("settings.json"):
        with open("settings.json") as f:
            keys = json.load(f)
            comet_apikey = keys.get("apikey")
            comet_username = keys.get("username")
            comet_restapikey = keys.get("restapikey")

    print("COMET_REST_API_KEY=%s" %(comet_restapikey))
    with open('.env', 'w') as writer:
        writer.write("COMET_API_KEY=%s\n" %(comet_apikey))
        writer.write("COMET_REST_API_KEY=%s\n" %(comet_restapikey))

    comet_api = API()
    return comet_api, comet_username, comet_project

def data_to_extract(username,args):
    labels = {}
    labels['title'] = "PPI Link Prediction"
    labels['x_label'] = "Iterations"
    labels['y_label'] = "Percent"
    if args.local:
        param_str = 'Local'
    else:
        param_str = 'Global'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.one_maml],\
                                   [args.two_maml],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.adamic_adar]\
                                   ]
    if args.local:
        labels['experiments_name'] = ['1-MAML','2-MAML', 'NoFinetune',\
                  'Finetune','Adamic-Adar']
    else:
        labels['experiments_name'] = ['1-MAML','2-MAML', 'Random', 'NoFinetune',\
              'Finetune']

    return labels

def data_to_extract_ppi(username,args):
    labels = {}
    labels['title'] = "PPI Link Prediction"
    labels['x_label'] = "Iterations"
    labels['y_label'] = "Percent"
    if args.local:
        param_str = 'Local'
    else:
        param_str = 'Global'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.two_maml],\
                                   [args.concat],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.adamic_adar],\
                                   [args.mlp],\
                                   [args.graph_sig]\
                                   ]
    if args.local:
        labels['experiments_name'] = ['2-MAML', '2-MAML-Concat','NoFinetune',\
                  'Finetune','Adamic-Adar','MLP', 'Inner-GraphSig']
    else:
        labels['experiments_name'] = ['2-MAML', '2-MAML-Concat', 'Random', 'NoFinetune',\
              'Finetune']
    args.local_block = 3
    args.global_block = [2]
    return labels

def data_to_extract_enzymes(username,args):
    labels = {}
    labels['title'] = "Enzymes Link Prediction"
    labels['x_label'] = "Iterations"
    labels['y_label'] = "Percent"
    if args.local:
        param_str = 'Local'
    else:
        param_str = 'Global'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.two_maml],\
                                   [args.concat],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.mlp],\
                                   [args.adamic_adar]\
                                   ]
    if args.local:
        labels['experiments_name'] = ['2-MAML', '2-MAML-Concat','NoFinetune',\
                  'Finetune','MLP','Adamic-Adar']
    else:
        labels['experiments_name'] = ['2-MAML', '2-MAML-Concat', 'Random', 'NoFinetune',\
              'Finetune']

    args.local_block = 3
    args.global_block = [2,6]
    return labels

def data_to_extract_reddit(username,args):
    labels = {}
    labels['title'] = "Reddit Link Prediction"
    labels['x_label'] = "Iterations"
    labels['y_label'] = "Percent"
    if args.local:
        param_str = 'Local_Batch'
    else:
        param_str = 'Global_Batch'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.two_maml],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.adamic_adar]\
                                   ]
    if args.local:
        labels['experiments_name'] = ['2-MAML', 'NoFinetune',\
                  'Finetune','Adamic-Adar']
    else:
        labels['experiments_name'] = ['2-MAML', 'Random', 'NoFinetune',\
              'Finetune']

    args.local_block = 3
    args.global_block = [2]
    return labels

def truncate_exp(data_experiments):
    last_data_points = [run[-1] for data_run in data_experiments for run in data_run]
    run_end_times = [timestep for timestep, value in last_data_points]
    earliest_end_time = min(run_end_times)

    clean_data_experiments = []
    for exp in data_experiments:
        clean_data_runs = []
        for run in exp:
            clean_data_runs.append({x: y for x, y in run if x <= earliest_end_time})
        clean_data_experiments.append(clean_data_runs)

    return clean_data_experiments

def get_data(args, title, x_label, y_label, labels_list, data, COMET_API_KEY,\
             COMET_REST_API_KEY,comet_username,comet_project):
    if not title or not x_label or not y_label or not labels_list:
        print("Error!!! Ensure filename, x and y labels,\
        and metric are present.")
        exit(1)

    train_auc = labels_list['train_metric_auc']
    train_ap = labels_list['train_metric_ap']
    test_auc = labels_list['test_metric_auc']
    test_ap = labels_list['test_metric_ap']

    comet_api, comet_username, comet_project = connect_to_comet(COMET_API_KEY,\
                                                                COMET_REST_API_KEY,\
                                                                comet_username,\
                                                                comet_project)

    # Accumulate data for all experiments.
    data_experiments_auc = []
    data_experiments_ap = []
    for i, runs in enumerate(data):
        # Accumulate data for all runs of a given experiment.
        if i >= args.local_block and not args.local:
            break
        if (i in args.global_block) and args.local:
            continue
        data_runs_auc = []
        data_runs_ap = []
        if len(runs) > 0:
            for exp_key in runs:
                try:
                    raw_data = comet_api.get("%s/%s/%s" %(comet_username,\
                                                            comet_project, exp_key))
                    if args.mode == 'Train':
                        for j in range(0,args.num_train_graphs):
                            metric_auc = train_auc + str(j) + "_AUC"
                            metric_ap = train_ap + str(j) + "_AP"
                            data_points_auc = raw_data.metrics_raw[metric_auc]
                            data_points_ap = raw_data.metrics_raw[metric_ap]
                            data_points_auc = [[point[0]+1,point[1]] for point in data_points_auc]
                            data_points_ap = [[point[0]+1,point[1]] for point in data_points_ap]
                            data_runs_auc.append(data_points_auc)
                            data_runs_ap.append(data_points_ap)
                    elif args.mode =='Test' and args.dataset =='Reddit':
                        for k in range(0,args.num_test_graphs):
                            metric_auc = test_auc + str(k) + "_AUC"
                            metric_ap = test_ap + str(k) + "_AP"
                            data_points_auc = raw_data.metrics_raw[metric_auc]
                            data_points_ap = raw_data.metrics_raw[metric_ap]
                            data_points_auc = [[point[0]+1,point[1]] for point in data_points_auc]
                            data_points_ap = [[point[0]+1,point[1]] for point in data_points_ap]
                            data_runs_auc.append(data_points_auc)
                            data_runs_ap.append(data_points_ap)
                    else:
                        for k in range(0,args.num_test_graphs):
                            metric_auc = test_auc + str(k) + "_AUC"
                            metric_ap = test_ap + str(k) + "_AP"
                            data_points_auc = raw_data.metrics_raw[metric_auc]
                            data_points_ap = raw_data.metrics_raw[metric_ap]
                            data_points_auc = [[point[0]+1,point[1]] for point in data_points_auc]
                            data_points_ap = [[point[0]+1,point[1]] for point in data_points_ap]
                            data_runs_auc.append(data_points_auc)
                            data_runs_ap.append(data_points_ap)

                    data_experiments_auc.append(data_runs_auc)
                    data_experiments_ap.append(data_runs_ap)
                except:
                    print("Failed on %s" %(exp_key))

    clean_data_experiments_auc = truncate_exp(data_experiments_auc)
    clean_data_experiments_ap = truncate_exp(data_experiments_ap)
    return clean_data_experiments_auc, clean_data_experiments_ap

def plot(**kwargs):
    labels = kwargs.get('labels')
    data = kwargs.get('data')

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()

    for label in (ax.get_xticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(20)
    for label in (ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.yticks(np.arange(0, 1, 0.1))
    ax.xaxis.get_offset_text().set_fontsize(10)
    axis_font = {'fontname': 'Arial', 'size': '24'}
    colors = sns.color_palette('colorblind', n_colors=len(data))

    # Plot data
    for runs, label, color in zip(data, labels.get('experiments_name'), colors):
        unique_x_values = set()
        for run in runs:
            for key in run.keys():
                unique_x_values.add(key)
        x_values = sorted(unique_x_values)

        # Plot mean and standard deviation of all runs
        y_values_mean = []
        y_values_std = []

        for x in x_values:
            y_values_mean.append(mean([run.get(x) for run in runs if run.get(x)]))
            y_values_std.append(np.std([run.get(x) for run in runs if run.get(x)]))

        x_values.insert(0,0)
        y_values_mean.insert(0,0)
        y_values_std.insert(0,0)
        print("%s average result after graphs %f" %(label,y_values_mean[-1]))
        # Plot std
        ax.fill_between(x_values, np.add(np.array(y_values_mean), np.array(y_values_std)),
                        np.subtract(np.array(y_values_mean), np.array(y_values_std)),
                        alpha=0.3,
                        edgecolor=color, facecolor=color)
        # Plot mean
        plt.plot(x_values, y_values_mean, color=color, linewidth=1.5, label=label)

    # Label figure
    ax.legend(loc='lower right', prop={'size': 16})
    ax.set_xlabel(labels.get('x_label'), **axis_font)
    ax.set_ylabel(labels.get('y_label'), **axis_font)
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)

    # remove grid lines
    ax.grid(False)
    plt.grid(b=False, color='w')
    return fig

def main(args):
    Joey_COMET_API_KEY="Ht9lkWvTm58fRo9ccgpabq5zV"
    Joey_COMET_REST_API_KEY="gvhm1m1y8OUTnPRqJarpeTapL"
    comet_project = args.comet_project
    comet_username = "joeybose"
    if args.dataset == 'PPI':
        extract_func = data_to_extract_ppi
    elif args.dataset == 'Reddit':
        comet_project = 'meta-graph-reddit'
        extract_func = data_to_extract_reddit

    labels = extract_func("joeybose",args)
    data_experiments_auc, data_experiments_ap = get_data(args,labels.get('title'), labels.get('x_label'),\
                                labels.get('y_label'), labels,\
                                labels.get('experiments_key'),COMET_API_KEY=Joey_COMET_API_KEY,\
                                COMET_REST_API_KEY=Joey_COMET_REST_API_KEY,\
                                comet_project=comet_project,\
                                comet_username=comet_username)
    fig_auc = plot(labels=labels, data=data_experiments_auc)
    fig_ap = plot(labels=labels, data=data_experiments_ap)
    if args.local:
        param_str = '_Local_'
    else:
        param_str = '_Global_'
    fig_auc.savefig('../plots_datasets/'+ args.dataset + '/' + args.file_str +
                    param_str + args.mode +'_new_AUC.png')
    fig_ap.savefig('../plots_datasets/'+ args.dataset + '/' + args.file_str +
                   param_str+ args.mode + '_new_AP.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_filename', default='plot_source.csv')
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument('--mode', type=str, default='Train')
    parser.add_argument('--file_str', type=str, default='')
    parser.add_argument('--one_maml', type=str, default='')
    parser.add_argument('--two_maml', type=str, default='')
    parser.add_argument('--concat', type=str, default='')
    parser.add_argument('--random_exp', type=str, default='')
    parser.add_argument('--no_finetune', type=str, default='')
    parser.add_argument('--finetune', type=str, default='')
    parser.add_argument('--adamic_adar', type=str, default='')
    parser.add_argument('--mlp', type=str, default='')
    parser.add_argument('--graph_sig', type=str, default='')
    parser.add_argument('--comet_project', type=str, default='meta-graph')
    parser.add_argument('--dataset', type=str, default='PPI')
    parser.add_argument('--local_block', type=int, default=0)
    parser.add_argument('--global_block', type=int, default=2)
    args = parser.parse_args()

    if args.dataset == 'PPI':
        args.num_train_graphs = 20
        args.num_test_graphs = 2
    elif args.dataset == 'ENZYMES':
        args.num_train_graphs = 10
        args.num_test_graphs = 10
    elif args.dataset == 'Reddit':
        args.num_train_graphs = 10
        args.num_test_graphs = 10
    else:
        raise NotImplementedError

    main(args)

