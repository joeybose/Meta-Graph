import json
import os
import ipdb
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import pickle
import joblib
from collections import Counter
from shutil import copyfile

data_path = '/home/joey.bose/dblp_papers_v11.txt'
save_path_base = '/home/joey.bose/aminer_data/'
save_path_rank_base = '/home/joey.bose/aminer_data_ranked/'

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == '__main__':
    """
    Rank Aminer-Citation v-11
    """
    fail_counter = 0
    onlyfiles = [f for f in listdir(save_path_base) if isfile(join(save_path_base, f))]
    file_dict = {}
    # Rank top 100 biggest files
    Topk = 100

    for i, file_ in tqdm(enumerate(onlyfiles),total=len(onlyfiles)):
        file_path = save_path_base + file_
        num_lines = file_len(file_path)
        file_dict[file_] = num_lines

    topk_files = sorted(file_dict.items(), key=lambda x:-x[1])[:Topk]
    print(topk_files)
    f = open("aminer_file_dict.pkl","wb")
    pickle.dump(file_dict,f)
    f.close()

    # Move Files
    if not os.path.exists(save_path_rank_base):
        os.mkdir(save_path_rank_base)

    for i, file_tuple in tqdm(enumerate(topk_files),total=Topk):
        file_name = file_tuple[0]
        src_path = save_path_base + file_name
        move_path = save_path_rank_base + file_name
        copyfile(src_path,move_path)
