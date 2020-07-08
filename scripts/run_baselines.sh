#!/usr/bin/env bash

set -x

echo "Getting into the script"

# No Finetuning
#python vgae.py --model='VGAE' --epochs=2500 --comet --meta_train_edge_ratio=0.1 --namestr='NoFinetune Ratio=0.1'
#python vgae.py --model='VGAE' --epochs=2500 --comet --meta_train_edge_ratio=0.2 --namestr='NoFinetune Ratio=0.2'
#python vgae.py --epochs=2500  --model='VGAE' --comet --meta_train_edge_ratio=0.3 --namestr='NoFinetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --comet --meta_train_edge_ratio=0.4 --namestr='NoFinetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --comet --meta_train_edge_ratio=0.5 --namestr='NoFinetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --comet --meta_train_edge_ratio=0.6 --namestr='NoFinetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --comet --meta_train_edge_ratio=0.7 --namestr='NoFinetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --comet --meta_train_edge_ratio=0.8 --namestr='NoFinetune Ratio=0.8'

## Finetuning
python vgae.py --epochs=2500 --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.1 --namestr='Finetune Ratio=0.1'
python vgae.py --epochs=2500 --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.2 --namestr='Finetune Ratio=0.2'
python vgae.py --epochs=2500  --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.3 --namestr='Finetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.4 --namestr='Finetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.5 --namestr='Finetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.6 --namestr='Finetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.7 --namestr='Finetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --finetune --comet --meta_train_edge_ratio=0.8 --namestr='Finetune Ratio=0.8'

# No Concat Finetuning
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.1 --namestr='Concat NoFinetune Ratio=0.1'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.2 --namestr='Concat NoFinetune Ratio=0.2'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.3 --namestr='Concat NoFinetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.4 --namestr=a'Concat NoFinetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.5 --namestr='Concat NoFinetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.6 --namestr='Concat NoFinetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.7 --namestr='Concat NoFinetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --comet --meta_train_edge_ratio=0.8 --namestr='Concat NoFinetune Ratio=0.8'

## Concat Finetuning
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.1 --namestr='Concat Finetune Ratio=0.1'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.2 --namestr='Concat Finetune Ratio=0.2'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.3 --namestr='Concat Finetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.4 --namestr='Concat Finetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.5 --namestr='Concat Finetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.6 --namestr='Concat Finetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.7 --namestr='Concat Finetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --concat_fixed_feats --finetune --comet --meta_train_edge_ratio=0.8 --namestr='Concat Finetune Ratio=0.8'

python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --comet --meta_train_edge_ratio=0.2 --namestr='ENZYMES NoFinetune Ratio=0.2'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --comet --meta_train_edge_ratio=0.3 --namestr='ENZYMES NoFinetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --comet --meta_train_edge_ratio=0.4 --namestr='ENZYMES NoFinetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --comet --meta_train_edge_ratio=0.5 --namestr='ENZYMES NoFinetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --comet --meta_train_edge_ratio=0.6 --namestr='ENZYMES NoFinetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --comet --meta_train_edge_ratio=0.7 --namestr='ENZYMES NoFinetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --comet --meta_train_edge_ratio=0.8 --namestr='ENZYMES NoFinetune Ratio=0.8'

# Finetuning
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.2 --namestr='ENZYMES Finetune Ratio=0.2'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.3 --namestr='ENZYMES Finetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.4 --namestr='ENZYMES Finetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.5 --namestr='ENZYMES Finetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.6 --namestr='ENZYMES Finetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.7 --namestr='ENZYMES Finetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.8 --namestr='ENZYMES Finetune Ratio=0.8'

#Concat feats
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --comet --meta_train_edge_ratio=0.2 --namestr='ENZYMES Concat NoFinetune Ratio=0.2'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --comet --meta_train_edge_ratio=0.3 --namestr='ENZYMES Concat NoFinetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --comet --meta_train_edge_ratio=0.4 --namestr='ENZYMES Concat NoFinetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --comet --meta_train_edge_ratio=0.5 --namestr='ENZYMES Concat NoFinetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --comet --meta_train_edge_ratio=0.6 --namestr='ENZYMES Concat NoFinetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --comet --meta_train_edge_ratio=0.7 --namestr='ENZYMES Concat NoFinetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --comet --meta_train_edge_ratio=0.8 --namestr='ENZYMES Concat NoFinetune Ratio=0.8'

# Finetuning
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.2 --namestr='ENZYMES Concat Finetune Ratio=0.2'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.3 --namestr='ENZYMES Concat Finetune Ratio=0.3'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.4 --namestr='ENZYMES Concat Finetune Ratio=0.4'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.5 --namestr='ENZYMES Concat Finetune Ratio=0.5'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.6 --namestr='ENZYMES Concat Finetune Ratio=0.6'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.7 --namestr='ENZYMES Concat Finetune Ratio=0.7'
python vgae.py --epochs=2500 --model='VGAE' --train_batch_size=4 --concat_fixed_feats --dataset=ENZYMES --finetune --comet --meta_train_edge_ratio=0.8 --namestr='ENZYMES Concat Finetune Ratio=0.8'
