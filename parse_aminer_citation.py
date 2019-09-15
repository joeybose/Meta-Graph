import json
import os
import csv
import pandas as pd
import ipdb
from tqdm import tqdm

data_path = '/home/joey.bose/dblp_papers_v11.txt'
save_path_base = '/home/joey.bose/aminer_data/'

def process_line(line, fail_counter):
    try:
        fos = data['fos']
        abstract = data['indexed_abstract']
        for field in fos:
            name = field['name']
            save_path = save_path_base + name.replace(" ", "") + '.txt'

            if not os.path.exists(save_path_base):
                os.mkdir(save_path_base)

            with open(save_path,"a+") as f:
                f.write(json.dumps(line))
                f.write('\n')
    except:
        fail_counter +=1
        if fail_counter % 100 ==0:
            print("Failed on a File | Total Fails %d" %(fail_counter))
    return fail_counter

if __name__ == '__main__':
    """
    Parse Aminer-Citation v-11
    """
    fail_counter = 0
    with open(data_path,'r', encoding="utf8") as f:
        for line in tqdm(f,total=4107340):
            data = json.loads(line)
            fail_counter = process_line(data,fail_counter)
