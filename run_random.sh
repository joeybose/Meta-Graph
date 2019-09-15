#!/usr/bin/env bash

set -x

echo "Getting into the script"
#echo "Running PPI experiments"

#python main.py --model='VGAE' --epochs=50 --meta_train_edge_ratio=0.1 --random_baseline --namestr='Random Baseline Ratio=0.1' --comet &
#python main.py --model='VGAE' --epochs=50 --meta_train_edge_ratio=0.2 --random_baseline --namestr='Random Baseline Ratio=0.2' --comet &
#python main.py --model='VGAE' --epochs=50 --meta_train_edge_ratio=0.3 --random_baseline --namestr='Random Baseline Ratio=0.3' --comet &
#python main.py --model='VGAE' --epochs=50 --meta_train_edge_ratio=0.4 --random_baseline --namestr='Random Baseline Ratio=0.4' --comet &
#python main.py --model='VGAE' --epochs=50 --meta_train_edge_ratio=0.5 --random_baseline --namestr='Random Baseline Ratio=0.5' --comet &
#python main.py --model='VGAE' --epochs=50 --meta_train_edge_ratio=0.6 --random_baseline --namestr='Random Baseline Ratio=0.6' --comet &
#python main.py --model='VGAE' --epochs=50 --meta_train_edge_ratio=0.7 --random_baseline --namestr='Random Baseline Ratio=0.7' --comet &

#echo "Running ENZYMES experiments"

#python main.py --model='VGAE' --epochs=50 --dataset=ENZYMES --meta_train_edge_ratio=0.1 --random_baseline --namestr='ENZYMES Random Baseline Ratio=0.1' --comet &
#python main.py --model='VGAE' --epochs=50 --dataset=ENZYMES --meta_train_edge_ratio=0.2 --random_baseline --namestr='ENZYMES Random Baseline Ratio=0.2' --comet &
#python main.py --model='VGAE' --epochs=50 --dataset=ENZYMES --meta_train_edge_ratio=0.3 --random_baseline --namestr='ENZYMES Random Baseline Ratio=0.3' --comet &
#python main.py --model='VGAE' --epochs=50 --dataset=ENZYMES --meta_train_edge_ratio=0.4 --random_baseline --namestr='ENZYMES Random Baseline Ratio=0.4' --comet &
#python main.py --model='VGAE' --epochs=50 --dataset=ENZYMES --meta_train_edge_ratio=0.5 --random_baseline --namestr='ENZYMES Random Baseline Ratio=0.5' --comet &
#python main.py --model='VGAE' --epochs=50 --dataset=ENZYMES --meta_train_edge_ratio=0.6 --random_baseline --namestr='ENZYMES Random Baseline Ratio=0.6' --comet &
#python main.py --model='VGAE' --epochs=50 --dataset=ENZYMES --meta_train_edge_ratio=0.7 --random_baseline --namestr='ENZYMES Random Baseline Ratio=0.7' --comet &

#wait

echo "Running REDDIT experiments"

python main.py --model='VGAE' --epochs=50 --dataset=REDDIT-MULTI-12K --use_fixed_feats --meta_train_edge_ratio=0.1 --random_baseline --namestr='REDDIT-MULTI-12K Random Baseline Ratio=0.1' --comet &
python main.py --model='VGAE' --epochs=50 --dataset=REDDIT-MULTI-12K --use_fixed_feats --meta_train_edge_ratio=0.2 --random_baseline --namestr='REDDIT-MULTI-12K Random Baseline Ratio=0.2' --comet &
python main.py --model='VGAE' --epochs=50 --dataset=REDDIT-MULTI-12K --use_fixed_feats --meta_train_edge_ratio=0.3 --random_baseline --namestr='REDDIT-MULTI-12K Random Baseline Ratio=0.3' --comet &
python main.py --model='VGAE' --epochs=50 --dataset=REDDIT-MULTI-12K --use_fixed_feats --meta_train_edge_ratio=0.4 --random_baseline --namestr='REDDIT-MULTI-12K Random Baseline Ratio=0.4' --comet &
python main.py --model='VGAE' --epochs=50 --dataset=REDDIT-MULTI-12K --use_fixed_feats --meta_train_edge_ratio=0.5 --random_baseline --namestr='REDDIT-MULTI-12K Random Baseline Ratio=0.5' --comet &
python main.py --model='VGAE' --epochs=50 --dataset=REDDIT-MULTI-12K --use_fixed_feats --meta_train_edge_ratio=0.6 --random_baseline --namestr='REDDIT-MULTI-12K Random Baseline Ratio=0.6' --comet &
python main.py --model='VGAE' --epochs=50 --dataset=REDDIT-MULTI-12K --use_fixed_feats --meta_train_edge_ratio=0.7 --random_baseline --namestr='REDDIT-MULTI-12K Random Baseline Ratio=0.7' --comet &
