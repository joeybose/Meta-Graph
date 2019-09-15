#!/usr/bin/env bash

set -x

echo "Getting into the script"

echo "AMINER 0.1"
### 0.1
python plot_wandb.py --mode='Train' --file_str='0.1' --two_maml='6r1yr00u' --no_finetune='4ivd72vi' --finetune='dxi5g89l' --mlp='x2le9exg' --graph_sig='32ovw30r'  --dataset='AMINER' --get_grad_steps
echo "AMINER 0.2"
### 0.2
python plot_wandb.py --mode='Train' --file_str='0.2' --two_maml='2evksgb8' --no_finetune='h1l3y9di' --finetune='wqd4c252 ' --mlp='zqdw5c0g' --graph_sig='32ovw30r'  --dataset='AMINER' --get_grad_steps
echo "AMINER 0.3"
### 0.3
python plot_wandb.py --mode='Train' --file_str='0.3' --two_maml='xefnxixv' --no_finetune='x1j83n08' --finetune='vd54v95d' --mlp='jrjl09cr' --graph_sig='mqhij7tm'  --dataset='AMINER' --get_grad_steps
echo "AMINER 0.4"
### 0.4
python plot_wandb.py --mode='Train' --file_str='0.4' --two_maml='5ny31pj1' --no_finetune='ivg4ymhc' --finetune='lq2xbxly' --mlp='jrjl09cr' --graph_sig='b5xbsxht'  --dataset='AMINER' --get_grad_steps
echo "AMINER 0.5"
### 0.5
python plot_wandb.py --mode='Train' --file_str='0.5' --two_maml='f6vjm2t1' --no_finetune='uc30rjw4' --finetune='uhzcx9tv' --mlp='la0af8s8' --graph_sig='a10djfgb'  --dataset='AMINER' --get_grad_steps
echo "AMINER 0.6"
### 0.6
python plot_wandb.py --mode='Train' --file_str='0.6' --two_maml='jmk7lwgb' --no_finetune='4wrbcf4t' --finetune='yf37ohc3' --mlp='pkec71vq' --graph_sig='21qnqbcz'  --dataset='AMINER' --get_grad_steps
echo "AMINER 0.7"
### 0.7
python plot_wandb.py --mode='Train' --file_str='0.7' --two_maml='yuyjl239' --no_finetune='n718az0j' --finetune='uovbbb2w' --mlp='sa0yq1q8' --graph_sig='pnj299hl'  --dataset='AMINER' --get_grad_steps
