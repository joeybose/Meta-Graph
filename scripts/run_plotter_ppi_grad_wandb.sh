#!/usr/bin/env bash

set -x

echo "Getting into the script"

echo "PPI 0.1"
### 0.1
python plot_wandb.py --mode='Train' --file_str='0.1' --two_maml='l5n28018' --concat='hjvwed7o' --no_finetune='x6w700c7' --finetune='15l49u26' --mlp='plu09evs' --graph_sig='o0pqonmb'  --dataset='PPI' --get_grad_steps
echo "PPI 0.2"
### 0.2
python plot_wandb.py --mode='Train' --file_str='0.2' --two_maml='665pydub' --concat='j9ia97gm' --no_finetune='dqvtbbf3' --finetune='tbs43bdn ' --mlp='3yxlpwbv' --graph_sig='8f6kukzs'  --dataset='PPI' --get_grad_steps
echo "PPI 0.3"
### 0.3
python plot_wandb.py --mode='Train' --file_str='0.3' --two_maml='azytfs4u' --concat='dip9z64b' --no_finetune='fusq205s' --finetune='zxhtln38' --mlp='fmmvh6s7' --graph_sig='pzocn7d6'  --dataset='PPI' --get_grad_steps
echo "PPI 0.4"
### 0.4
python plot_wandb.py --mode='Train' --file_str='0.4' --two_maml='cmyavyxz' --concat='1vhfzajp' --no_finetune='c3tl70d3' --finetune='v8dt60jw' --mlp='z6vc0iql' --graph_sig='csl5vhy7'  --dataset='PPI' --get_grad_steps
echo "PPI 0.5"
### 0.5
python plot_wandb.py --mode='Train' --file_str='0.5' --two_maml='mgrs9sss' --concat='ot4c8qm4' --no_finetune='aam4zuim' --finetune='35e8mecc' --mlp='ohrk5bzd' --graph_sig='ww7z5tsf'  --dataset='PPI' --get_grad_steps
echo "PPI 0.6"
### 0.6
python plot_wandb.py --mode='Train' --file_str='0.6' --two_maml='c8qqc6ti' --concat='jn8kw9pn' --no_finetune='zg5xlims' --finetune='lowk3n6j' --mlp='i5fgp3eh' --graph_sig='mtlffk3p'  --dataset='PPI' --get_grad_steps
echo "PPI 0.7"
### 0.7
python plot_wandb.py --mode='Train' --file_str='0.7' --two_maml='my3soycv' --concat='jbkm9yoj' --no_finetune='4n4t5xmp' --finetune='tplubzay' --mlp='mqskgi0m' --graph_sig='amvq1kn7'  --dataset='PPI' --get_grad_steps
