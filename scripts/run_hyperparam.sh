#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running Batch Size sweep PPI experiments"

python main.py --meta_train_edge_ratio=0.2 --epochs=10 --train_batch_size=1 --model='VGAE' --order=2 --namestr='BS 1 2-MAML PPI Ratio=0.2' --comet
python main.py --meta_train_edge_ratio=0.2 --epochs=10 --train_batch_size=2 --model='VGAE' --order=2 --namestr='BS 2 2-MAML PPI Ratio=0.2' --comet
python main.py --meta_train_edge_ratio=0.2 --epochs=10 --train_batch_size=4 --model='VGAE' --order=2 --namestr='BS 4 2-MAML PPI Ratio=0.2' --comet
python main.py --meta_train_edge_ratio=0.2 --epochs=10 --train_batch_size=5 --model='VGAE' --order=2 --namestr='BS 5 2-MAML PPI Ratio=0.2' --comet
python main.py --meta_train_edge_ratio=0.2 --epochs=10 --train_batch_size=10 --model='VGAE' --order=2 --namestr='BS 10 2-MAML PPI Ratio=0.2' --comet
