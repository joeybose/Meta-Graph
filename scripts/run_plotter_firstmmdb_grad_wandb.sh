#!/usr/bin/env bash

set -x

echo "Getting into the script"

echo "PPI 0.1"
### 0.1
python plot_wandb.py --mode='Train' --file_str='0.1' --two_maml='28oseu07' --no_finetune='s96dkq1z' --finetune='df8e2mhh' --mlp='gm0z3yxd' --graph_sig='bs0c1glm'  --dataset='FIRSTMM_DB' --get_grad_steps
echo "PPI 0.2"
### 0.2
python plot_wandb.py --mode='Train' --file_str='0.2' --two_maml='g6k3381h' --no_finetune='jpsfaism' --finetune='vowc1u48' --mlp='cgvz7534' --graph_sig='o8ewiyol'  --dataset='FIRSTMM_DB' --get_grad_steps
echo "PPI 0.3"
### 0.3
python plot_wandb.py --mode='Train' --file_str='0.3' --two_maml='0lo19msp' --no_finetune='tlnxol4d' --finetune='nad2ex1q' --mlp='joqd1918' --graph_sig='c4t93v8h'  --dataset='FIRSTMM_DB' --get_grad_steps
echo "PPI 0.4"
### 0.4
python plot_wandb.py --mode='Train' --file_str='0.4' --two_maml='c8d28g6x' --no_finetune='tfincgtq' --finetune='o53cszvf' --mlp='6absh1lw' --graph_sig='cq9wlh5n'  --dataset='FIRSTMM_DB' --get_grad_steps
echo "PPI 0.5"
### 0.5
python plot_wandb.py --mode='Train' --file_str='0.5' --two_maml='pahhhnd2' --no_finetune='0dg1jrid' --finetune='bipbpuqx' --mlp='z38ximym' --graph_sig='5gdfdlys'  --dataset='FIRSTMM_DB' --get_grad_steps
echo "PPI 0.6"
### 0.6
python plot_wandb.py --mode='Train' --file_str='0.6' --two_maml='gwokit1o' --no_finetune='ud6hpkp5' --finetune='y5c191dh' --mlp='x50hlrep' --graph_sig='u1g20x5n'  --dataset='FIRSTMM_DB' --get_grad_steps
echo "PPI 0.7"
### 0.7
python plot_wandb.py --mode='Train' --file_str='0.7' --two_maml='7weacjzd' --no_finetune='ajwxpznq' --finetune='deyr9bj7' --mlp='x7j9c5az' --graph_sig='wpqy10x1'  --dataset='FIRSTMM_DB' --get_grad_steps
