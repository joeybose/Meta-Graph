### Meta-Learning on Graphs ###
This repository contains code for the arXiv Preprint:
"Link Prediction from Sparse DataUsing Meta Learning"
by: Avishek Joey Bose, Ankit Jain, Piero Molino, William L. Hamilton

ArXiv Link: https://arxiv.org/abs/1912.09867
If this repository is helpful in your research, please consider citing us.

```
@article{bose2019meta,
  title={Meta-Graph: Few Shot Link Prediction via Meta Learning},
  author={Bose, Avishek Joey and Jain, Ankit and Molino, Piero and Hamilton, William L},
  journal={arXiv preprint arXiv:1912.09867},
  year={2019}
}
```

Some Requirements:
- pytorch geometric
- scikit-learn==0.22
- comet_ml
- wandb
- grakel
- torchviz

This codebase has many different flags so its important one familiarizes themselves with all the command line args.
The easiest accesspoint to the codebase is using some prepared scripts in the scripts folder.

Here are some sample commands:

## Running Graph Signature on PPI
`python3 main.py --meta_train_edge_ratio=0.1 --model='VGAE'
--encoder='GraphSignature' --epochs=46 --use_gcn_sig --concat_fixed_feats
--inner_steps=2 --inner-lr=2.24e-3 --meta-lr=2.727e-3 --clip_grad
--patience=2000 --train_batch_size=1 --dataset=PPI --order=2
--namestr='2-MAML_Concat_Patience_Best_GS_PPI_Ratio=0.1'`

This command will run the Meta-Graph algorithm using 10% training edges for each graph.
It will also use the default GraphSignature function as the encoder in a VGAE. The `--use_gcn_sig`
flag will force the GraphSignature to use a GCN style signature function  and finally
order 2 will perform second order optimization.
