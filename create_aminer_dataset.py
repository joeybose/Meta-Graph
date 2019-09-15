import json
import os
import ipdb
from tqdm import tqdm
import argparse
from os import listdir
from os.path import isfile, join
import pickle
import joblib
from collections import Counter
from shutil import copyfile
import networkx as nx
import spacy
import nltk
import numpy as np

nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
data_path = '/home/joey.bose/dblp_papers_v11.txt'
save_path_base = '/home/joey.bose/aminer_data/'
load_path_rank_base = '/home/joey.bose/aminer_data_ranked/fos/'
save_path_graph_base = '/home/joey.bose/aminer_data_ranked/graphs/'
raw_save_path = '/home/joey.bose/aminer_data_ranked/aminer_raw.txt'
spacy_nlp = spacy.load('en_core_web_sm')
glove_path = '/home/joey.bose/docker_temp/meta-graph/meta-graph/glove.840B.300d.txt'

class Lang:
    def __init__(self):
        super(Lang, self).__init__()
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def gen_embeddings(vocab, file, emb_size, emb_dim):
    """
    Generate an initial embedding matrix for word_dict.
    If an embedding file is not given or a word is not in the embedding file,
    a randomly initialized vector will be used.
    """
    # embeddings = np.random.randn(vocab.n_words, emb_size) * 0.01
    embeddings = np.zeros((vocab.n_words, emb_size))
    print('Embeddings: %d x %d' % (vocab.n_words, emb_size))
    if file is not None:
        print('Loading embedding file: %s' % file)
        pre_trained = 0
        for line in open(file).readlines():
            sp = line.split()
            if(len(sp) == emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

def process_raw_abstracts(vocab):
    with open(raw_save_path,"r",encoding="utf8") as f:
        for line in tqdm(f,total=13304586):
            tokens = nltk.tokenize.word_tokenize(line)
            tokens = [token for token in tokens if not token in nltk_stopwords]
            vocab.index_words(tokens)

def get_node_embed(text,vocab):
    sum_embed = 0
    for word in text:
        embed = embeddings[vocab.word2index[word]]
        sum_embed += embed
    return sum_embed
def check_graph(G):
    total_nodes = len(G.nodes)
    no_emb_nodes = 0
    nodes_to_delete = []
    for node_str in G.nodes:
        try:
            emb = G.node[node_str]['emb']
        except:
            no_emb_nodes += 1
            nodes_to_delete.append(node_str)
    print("%d Nodes and %d missing nodes in G " %(total_nodes, no_emb_nodes))
    G.remove_nodes_from(nodes_to_delete)
    return G

def process_line(G, line, vocab=None):
    try:
        fos = data['fos']
        abstract = data['indexed_abstract']
        paper_id = data['id']
        references_id = data['references']
        text = list(abstract['InvertedIndex'].keys())
        text =" ".join(text)
        if args.process_raw:
            with open(raw_save_path,"a+") as f:
                f.write(text)
                f.write('\n')

        '''Create Node Embedding if Node doesn't exist '''
        if vocab is not None:
            tokens = nltk.tokenize.word_tokenize(text)
            tokens = [token for token in tokens if not token in nltk_stopwords]
            node_emb = get_node_embed(tokens,vocab)

        for field in fos:
            name = field['name']
        for ref in references_id:
            G.add_edge(paper_id, ref)
            G.node[paper_id]['emb'] = node_emb
    except:
        return G

    return G

if __name__ == '__main__':
    """
    Create Aminer-Citation v-11 Graphs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default='100')
    parser.add_argument("--process_raw", action="store_true", default=False,
		help='Process Raw Data')
    parser.add_argument("--make_vocab", action="store_true", default=False,
		help='Create Vocab from the raw abstract data')
    args = parser.parse_args()
    onlyfiles = [f for f in listdir(load_path_rank_base) if isfile(join(load_path_rank_base, f))]
    vocab = Lang()
    if args.make_vocab:
        process_raw_abstracts(vocab)
        joblib.dump(vocab, "aminer_100_vocab.pkl")
        print("Done generating vocab")
        embeddings = gen_embeddings(vocab,file=glove_path,emb_size=300,emb_dim=300)
        joblib.dump(embeddings, "aminer_100_embed.pkl")
        print("Done")
        exit()
    else:
        vocab = joblib.load("aminer_100_vocab.pkl")
        embeddings = joblib.load("aminer_100_embed.pkl")

    for i, file_ in tqdm(enumerate(onlyfiles),total=len(onlyfiles)):
        file_path = load_path_rank_base + file_
        G = nx.Graph()
        with open(file_path,'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                G = process_line(G,data,vocab)
        G = check_graph(G)
        print("%s has %d Nodes and %d edges" %(file_,len(G),len(G.edges)))
        if not os.path.exists(save_path_graph_base):
            os.mkdir(save_path_graph_base)
        save_path_graph = save_path_graph_base + file_.split('.')[0] + '_graph.pkl'
        nx.write_gpickle(G,save_path_graph)

