#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running PPI experiments"

#python main.py --meta_train_edge_ratio=0.1 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.1' --comet
#python main.py --meta_train_edge_ratio=0.2 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.2' --comet
#python main.py --meta_train_edge_ratio=0.3 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.3' --comet
#python main.py --meta_train_edge_ratio=0.4 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.4' --comet
#python main.py --meta_train_edge_ratio=0.5 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.5' --comet
#python main.py --meta_train_edge_ratio=0.6 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.6' --comet
#python main.py --meta_train_edge_ratio=0.7 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.7' --comet
#python main.py --meta_train_edge_ratio=0.8 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML PPI Ratio=0.8' --comet

#python main.py --meta_train_edge_ratio=0.1 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML MLP PPI Ratio=0.1' --comet
#python main.py --meta_train_edge_ratio=0.2 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML MLP PPI Ratio=0.2' --comet
#python main.py --meta_train_edge_ratio=0.3 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML MLP PPI Ratio=0.3' --comet
#python main.py --meta_train_edge_ratio=0.4 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML MLP PPI Ratio=0.4' --comet
#python main.py --meta_train_edge_ratio=0.5 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML MLP PPI Ratio=0.5' --comet
#python main.py --meta_train_edge_ratio=0.6 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML MLP PPI Ratio=0.6' --comet
#python main.py --meta_train_edge_ratio=0.7 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML MLP PPI Ratio=0.7' --comet

echo "Running PPI Graph Signature experiments"
python main.py --meta_train_edge_ratio=0.1 --model='VGAE' --encoder='GraphSignature' --inner-lr=1e-3 --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML GS PPI Ratio=0.1' --comet
python main.py --meta_train_edge_ratio=0.2 --model='VGAE' --encoder='GraphSignature' --inner-lr=1e-3 --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML GS PPI Ratio=0.2' --comet
python main.py --meta_train_edge_ratio=0.3 --model='VGAE' --encoder='GraphSignature' --inner-lr=1e-3 --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML GS PPI Ratio=0.3' --comet
python main.py --meta_train_edge_ratio=0.4 --model='VGAE' --encoder='GraphSignature' --inner-lr=1e-3 --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML GS PPI Ratio=0.4' --comet
python main.py --meta_train_edge_ratio=0.5 --model='VGAE' --encoder='GraphSignature' --inner-lr=1e-3 --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML GS PPI Ratio=0.5' --comet
python main.py --meta_train_edge_ratio=0.6 --model='VGAE' --encoder='GraphSignature' --inner-lr=1e-3 --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML GS PPI Ratio=0.6' --comet
python main.py --meta_train_edge_ratio=0.7 --model='VGAE' --encoder='GraphSignature' --inner-lr=1e-3 --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML GS PPI Ratio=0.7' --comet

echo "Running PPI with concat-feats experiments"

#python main.py --meta_train_edge_ratio=0.1 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --order=2 --namestr='2-MAML Concat PPI Ratio=0.1' --comet
#python main.py --meta_train_edge_ratio=0.2 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --order=2 --namestr='2-MAML Concat PPI Ratio=0.2' --comet
#python main.py --meta_train_edge_ratio=0.3 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --order=2 --namestr='2-MAML Concat PPI Ratio=0.3' --comet
#python main.py --meta_train_edge_ratio=0.4 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --order=2 --namestr='2-MAML Concat PPI Ratio=0.4' --comet
#python main.py --meta_train_edge_ratio=0.5 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --order=2 --namestr='2-MAML Concat PPI Ratio=0.5' --comet
#python main.py --meta_train_edge_ratio=0.6 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --order=2 --namestr='2-MAML Concat PPI Ratio=0.6' --comet
#python main.py --meta_train_edge_ratio=0.7 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --order=2 --namestr='2-MAML Concat PPI Ratio=0.7' --comet

echo "Running ENZYMES experiments"

#python main.py --meta_train_edge_ratio=0.2 --model='VGAE' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML ENZYMES Ratio=0.2' --comet
#python main.py --meta_train_edge_ratio=0.3 --model='VGAE' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML ENZYMES Ratio=0.3' --comet
#python main.py --meta_train_edge_ratio=0.4 --model='VGAE' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML ENZYMES Ratio=0.4' --comet
#python main.py --meta_train_edge_ratio=0.5 --model='VGAE' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML ENZYMES Ratio=0.5' --comet
#python main.py --meta_train_edge_ratio=0.6 --model='VGAE' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML ENZYMES Ratio=0.6' --comet
#python main.py --meta_train_edge_ratio=0.7 --model='VGAE' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML ENZYMES Ratio=0.7' --comet


echo "Running ENZYMES with concat-feats experiments"

#python main.py --meta_train_edge_ratio=0.2 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML Concat ENZYMES Ratio=0.2' --comet
#python main.py --meta_train_edge_ratio=0.3 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML Concat ENZYMES Ratio=0.3' --comet
#python main.py --meta_train_edge_ratio=0.4 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML Concat ENZYMES Ratio=0.4' --comet
#python main.py --meta_train_edge_ratio=0.5 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML Concat ENZYMES Ratio=0.5' --comet
#python main.py --meta_train_edge_ratio=0.6 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML Concat ENZYMES Ratio=0.6' --comet
#python main.py --meta_train_edge_ratio=0.7 --model='VGAE' --epochs=50 --concat_fixed_feats --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML Concat ENZYMES Ratio=0.7' --comet

echo "Running ENZYMES MLP experiments"

#python main.py --meta_train_edge_ratio=0.2 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML MLP ENZYMES Ratio=0.2' --comet
#python main.py --meta_train_edge_ratio=0.3 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML MLP ENZYMES Ratio=0.3' --comet
#python main.py --meta_train_edge_ratio=0.4 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML MLP ENZYMES Ratio=0.4' --comet
#python main.py --meta_train_edge_ratio=0.5 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML MLP ENZYMES Ratio=0.5' --comet
#python main.py --meta_train_edge_ratio=0.6 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML MLP ENZYMES Ratio=0.6' --comet
#python main.py --meta_train_edge_ratio=0.7 --model='VGAE' --encoder='MLP' --epochs=50 --train_batch_size=1 --dataset=ENZYMES --order=2 --namestr='2-MAML MLP ENZYMES Ratio=0.7' --comet

echo "Running ENZYMES Random Baseline experiments"

#python main.py --epochs=50 --meta_train_edge_ratio=0.1 --dataset=ENZYMES --random_baseline --namestr='Random ENZYMES Baseline Ratio=0.1' --comet
#python main.py --epochs=50 --meta_train_edge_ratio=0.2 --dataset=ENZYMES --random_baseline --namestr='Random ENZYMES Baseline Ratio=0.2' --comet
#python main.py --epochs=50 --meta_train_edge_ratio=0.3 --dataset=ENZYMES --random_baseline --namestr='Random ENZYMES Baseline Ratio=0.3' --comet
#python main.py --epochs=50 --meta_train_edge_ratio=0.4 --dataset=ENZYMES --random_baseline --namestr='Random ENZYMES Baseline Ratio=0.4' --comet
#python main.py --epochs=50 --meta_train_edge_ratio=0.5 --dataset=ENZYMES --random_baseline --namestr='Random ENZYMES Baseline Ratio=0.5' --comet
#python main.py --epochs=50 --meta_train_edge_ratio=0.6 --dataset=ENZYMES --random_baseline --namestr='Random ENZYMES Baseline Ratio=0.6' --comet
#python main.py --epochs=50 --meta_train_edge_ratio=0.7 --dataset=ENZYMES --random_baseline --namestr='Random ENZYMES Baseline Ratio=0.7' --comet

echo "Running ENZYMES Adamic Baseline experiments"

#python  main.py --meta_train_edge_ratio=0.1 --model='VGAE' --epochs=50 --dataset=ENZYMES --train_batch_size=1 --order=2 --namestr='2-MAML ENZYMES Adamic Ratio=0.1' --adamic_adar_baseline --comet
#python  main.py --meta_train_edge_ratio=0.2 --model='VGAE' --epochs=50 --dataset=ENZYMES --train_batch_size=1 --order=2 --namestr='2-MAML ENZYMES Adamic Ratio=0.2' --adamic_adar_baseline --comet
#python  main.py --meta_train_edge_ratio=0.3 --model='VGAE' --epochs=50 --dataset=ENZYMES --train_batch_size=1 --order=2 --namestr='2-MAML ENZYMES Adamic Ratio=0.3' --adamic_adar_baseline --comet
#python  main.py --meta_train_edge_ratio=0.4 --model='VGAE' --epochs=50 --dataset=ENZYMES --train_batch_size=1 --order=2 --namestr='2-MAML ENZYMES Adamic Ratio=0.4' --adamic_adar_baseline --comet
#python  main.py --meta_train_edge_ratio=0.5 --model='VGAE' --epochs=50 --dataset=ENZYMES --train_batch_size=1 --order=2 --namestr='2-MAML ENZYMES Adamic Ratio=0.5' --adamic_adar_baseline --comet
#python  main.py --meta_train_edge_ratio=0.6 --model='VGAE' --epochs=50 --dataset=ENZYMES --train_batch_size=1 --order=2 --namestr='2-MAML ENZYMES Adamic Ratio=0.6' --adamic_adar_baseline --comet
#python  main.py --meta_train_edge_ratio=0.7 --model='VGAE' --epochs=50 --dataset=ENZYMES --train_batch_size=1 --order=2 --namestr='2-MAML ENZYMES Adamic Ratio=0.7' --adamic_adar_baseline --comet









