#!/usr/bin/env bash

set -x

echo "Getting into the script"

python  main.py --meta_train_edge_ratio=0.1 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML Adamic PPI Ratio=0.1' --adamic_adar_baseline --comet
python  main.py --meta_train_edge_ratio=0.2 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML Adamic PPI Ratio=0.2' --adamic_adar_baseline --comet
python  main.py --meta_train_edge_ratio=0.3 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML Adamic PPI Ratio=0.3' --adamic_adar_baseline --comet
python  main.py --meta_train_edge_ratio=0.4 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML Adamic PPI Ratio=0.4' --adamic_adar_baseline --comet
python  main.py --meta_train_edge_ratio=0.5 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML Adamic PPI Ratio=0.5' --adamic_adar_baseline --comet
python  main.py --meta_train_edge_ratio=0.6 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML Adamic PPI Ratio=0.6' --adamic_adar_baseline --comet
python  main.py --meta_train_edge_ratio=0.7 --model='VGAE' --epochs=50 --train_batch_size=1 --order=2 --namestr='2-MAML Adamic PPI Ratio=0.7' --adamic_adar_baseline --comet
