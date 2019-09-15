#!/usr/bin/env bash

set -x

echo "Getting into the script"
python3 main.py --meta_train_edge_ratio=0.1 --model='VGAE' --encoder='GraphSignature' --epochs=46 --use_gcn_sig --concat_fixed_feats --inner_steps=2 --inner-lr=2.24e-3 --meta-lr=2.727e-3 --clip_grad --patience=2000 --train_batch_size=1 --dataset=PPI --order=2 --namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.1' --comet --wandb
python3 main.py --meta_train_edge_ratio=0.2 --model='VGAE' --encoder='GraphSignature' --epochs=32 --use_gcn_sig --concat_fixed_feats --inner_steps=23 --inner-lr=4.949e-2 --meta-lr=2.834e-3 --clip_grad --patience=2000 --train_batch_size=1 --dataset=PPI --order=2 --namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.2' --comet --wandb
python3 main.py --meta_train_edge_ratio=0.3 --model='VGAE' --encoder='GraphSignature' --epochs=39 --use_gcn_sig --concat_fixed_feats --inner_steps=15 --inner-lr=3.545e-3 --meta-lr=1.493e-2 --clip_grad --patience=2000 --train_batch_size=1 --dataset=PPI --order=2 --namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.3' --comet --wandb
python3 main.py --meta_train_edge_ratio=0.4 --model='VGAE' --encoder='GraphSignature' --epochs=36 --use_gcn_sig --concat_fixed_feats --inner_steps=30 --inner-lr=8.618e-4 --meta-lr=1.1192e-2 --clip_grad --patience=2000 --train_batch_size=1 --dataset=PPI --order=2 --namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.4' --comet --wandb
python3 main.py --meta_train_edge_ratio=0.5 --model='VGAE' --encoder='GraphSignature' --epochs=7 --use_gcn_sig --concat_fixed_feats --inner_steps=15 --inner-lr=6.07e-3 --meta-lr=1.337e-2 --clip_grad --patience=2000 --train_batch_size=1 --dataset=PPI --order=2 --namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.5' --comet --wandb
python3 main.py --meta_train_edge_ratio=0.6 --model='VGAE' --encoder='GraphSignature' --epochs=36 --use_gcn_sig --concat_fixed_feats --inner_steps=9 --inner-lr=4.14e-4 --meta-lr=1.42e-3 --clip_grad --patience=2000 --train_batch_size=1 --dataset=PPI --order=2 --namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.6' --comet --wandb
python3 main.py --meta_train_edge_ratio=0.7 --model='VGAE' --encoder='GraphSignature' --epochs=18 --use_gcn_sig --concat_fixed_feats --inner_steps=29 --inner-lr=2.592e-2 --meta-lr=1.729e-3 --clip_grad --patience=2000 --train_batch_size=1 --dataset=PPI --order=2 --namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.7' --comet --wandb
