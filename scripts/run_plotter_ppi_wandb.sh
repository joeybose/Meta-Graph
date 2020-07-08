#!/usr/bin/env bash

set -x

echo "Getting into the script"

### 0.1
python plot_wandb.py --mode='Train' --file_str='0.1' --two_maml='51a137h0' --random_exp='958jmtdt' --concat='5h6tv117' --no_finetune='oicast9p' --finetune='jwtqqcnx' --adamic_adar='jym8wrdk' --mlp='ycict122' --graph_sig='fra1kxzs' --graph_sig_concat='f5n0421a' --graph_sig_random='acwam7wd' --dataset='PPI' --no_bar_plot --global_block 3 4 5
python plot_wandb.py --mode='Test' --file_str='0.1' --two_maml='51a137h0' --random_exp='958jmtdt' --concat='5h6tv117' --no_finetune='oicast9p' --finetune='jwtqqcnx' --adamic_adar='jym8wrdk' --mlp='ycict122' --graph_sig='fra1kxzs' --graph_sig_concat='f5n0421a' --graph_sig_random='acwam7wd' --dataset='PPI' --no_bar_plot --global_block 3 4 5
python plot_wandb.py --local --mode='Train' --file_str='0.1' --two_maml='51a137h0' --random_exp='958jmtdt' --concat='5h6tv117' --no_finetune='oicast9p' --finetune='jwtqqcnx' --adamic_adar='jym8wrdk' --mlp='ycict122' --graph_sig='fra1kxzs' --graph_sig_concat='f5n0421a' --graph_sig_random='acwam7wd' --dataset='PPI' --no_bar_plot --global_block 3 4 5
python plot_wandb.py --local --mode='Test' --file_str='0.1' --two_maml='51a137h0' --random_exp='958jmtdt' --concat='5h6tv117' --no_finetune='oicast9p' --finetune='jwtqqcnx' --adamic_adar='jym8wrdk' --mlp='ycict122' --graph_sig='fra1kxzs' --graph_sig_concat='f5n0421a' --graph_sig_random='acwam7wd' --dataset='PPI' --no_bar_plot --global_block 3 4 5

#### 0.2
#python plot_wandb.py --mode='Train' --file_str='0.2' --two_maml='2fx6yzar' --random_exp='r5nf1r3g' --concat='nwrxg66k' --no_finetune='mfd3suff' --finetune='ufpmxpe7' --adamic_adar='xk73qql6' --mlp='ycm7128c' --graph_sig='18z1zlhp' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --mode='Test' --file_str='0.2' --two_maml='2fx6yzar' --random_exp='r5nf1r3g' --concat='nwrxg66k' --no_finetune='mfd3suff' --finetune='ufpmxpe7' --adamic_adar='xk73qql6' --mlp='ycm7128c' --graph_sig='18z1zlhp' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Train' --file_str='0.2' --two_maml='2fx6yzar' --random_exp='r5nf1r3g' --concat='nwrxg66k' --no_finetune='mfd3suff' --finetune='ufpmxpe7' --adamic_adar='xk73qql6' --mlp='ycm7128c' --graph_sig='18z1zlhp' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Test' --file_str='0.2' --two_maml='2fx6yzar' --random_exp='r5nf1r3g' --concat='nwrxg66k' --no_finetune='mfd3suff' --finetune='ufpmxpe7' --adamic_adar='xk73qql6' --mlp='ycm7128c' --graph_sig='18z1zlhp' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5

##### 0.3
#python plot_wandb.py --mode='Train' --file_str='0.3' --two_maml='m5fxvkfv' --random_exp='so4007b8' --concat='icvqm55w' --no_finetune='drusjtez' --finetune='b1qbrhga' --adamic_adar='le29sm1i' --mlp='th44x964' --graph_sig='2se6o3nq' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --mode='Test' --file_str='0.3' --two_maml='m5fxvkfv' --random_exp='so4007b8' --concat='icvqm55w' --no_finetune='drusjtez' --finetune='b1qbrhga' --adamic_adar='le29sm1i' --mlp='th44x964' --graph_sig='2se6o3nq' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Train' --file_str='0.3' --two_maml='m5fxvkfv' --random_exp='so4007b8' --concat='icvqm55w' --no_finetune='drusjtez' --finetune='b1qbrhga' --adamic_adar='le29sm1i' --mlp='th44x964' --graph_sig='2se6o3nq' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Test' --file_str='0.3' --two_maml='m5fxvkfv' --random_exp='so4007b8' --concat='icvqm55w' --no_finetune='drusjtez' --finetune='b1qbrhga' --adamic_adar='le29sm1i' --mlp='th44x964' --graph_sig='2se6o3nq' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5

##### 0.4
#python plot_wandb.py --mode='Train' --file_str='0.4' --two_maml='kiin2ltu' --random_exp='nzqt7x0m' --concat='1qh49vh0' --no_finetune='3ogmputq' --finetune='vezexdms' --adamic_adar='k3d9t34j' --mlp='43n81tbp' --graph_sig='eejt9gu9' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --mode='Test' --file_str='0.4' --two_maml='kiin2ltu' --random_exp='nzqt7x0m' --concat='1qh49vh0' --no_finetune='3ogmputq' --finetune='vezexdms' --adamic_adar='k3d9t34j' --mlp='43n81tbp' --graph_sig='eejt9gu9' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Train' --file_str='0.4' --two_maml='kiin2ltu' --random_exp='nzqt7x0m' --concat='1qh49vh0' --no_finetune='3ogmputq' --finetune='vezexdms' --adamic_adar='k3d9t34j' --mlp='43n81tbp' --graph_sig='eejt9gu9' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Test' --file_str='0.4' --two_maml='kiin2ltu' --random_exp='nzqt7x0m' --concat='1qh49vh0' --no_finetune='3ogmputq' --finetune='vezexdms' --adamic_adar='k3d9t34j' --mlp='43n81tbp' --graph_sig='eejt9gu9' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5

##### 0.5
##python plot_wandb.py --mode='Train' --file_str='0.5' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4
##python plot_wandb.py --mode='Test' --file_str='0.5' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4
##python plot_wandb.py --local --mode='Train' --file_str='0.5' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4
##python plot_wandb.py --local --mode='Test' --file_str='0.5' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4

##### 0.6
##python plot_wandb.py --mode='Train' --file_str='0.6' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4
##python plot_wandb.py --mode='Test' --file_str='0.6' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4
##python plot_wandb.py --local --mode='Train' --file_str='0.6' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4
##python plot_wandb.py --local --mode='Test' --file_str='0.6' --two_maml='24q2qh3e' --random_exp='' --no_finetune='vn2xjqev' --finetune='a4sk262k' --adamic_adar='vlv2mv1t' --mlp='17llheum' --graph_sig='fra1kxzs' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 1 --global_block 1 2 3 4

#### 0.7
#python plot_wandb.py --mode='Train' --file_str='0.7' --two_maml='12xe6dux' --random_exp='xga4kp7j' --concat='q78a3qsr' --no_finetune='pr5xaguf' --finetune='op2llf77' --adamic_adar='yhnq5i55' --mlp='3f541vpt' --graph_sig='m583zz8y' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --mode='Test' --file_str='0.7' --two_maml='12xe6dux' --random_exp='xga4kp7j' --concat='q78a3qsr' --no_finetune='pr5xaguf' --finetune='op2llf77' --adamic_adar='yhnq5i55' --mlp='3f541vpt' --graph_sig='m583zz8y' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Train' --file_str='0.7' --two_maml='12xe6dux' --random_exp='xga4kp7j' --concat='q78a3qsr' --no_finetune='pr5xaguf' --finetune='op2llf77' --adamic_adar='yhnq5i55' --mlp='3f541vpt' --graph_sig='m583zz8y' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
#python plot_wandb.py --local --mode='Test' --file_str='0.7' --two_maml='12xe6dux' --random_exp='xga4kp7j' --concat='q78a3qsr' --no_finetune='pr5xaguf' --finetune='op2llf77' --adamic_adar='yhnq5i55' --mlp='3f541vpt' --graph_sig='m583zz8y' --graph_sig_concat='' --graph_sig_random='' --dataset='PPI' --local_block 2 --global_block 3 4 5
