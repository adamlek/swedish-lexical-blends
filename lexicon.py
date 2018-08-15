import pickle
from collections import defaultdict
from helper_functions import format_lemma, get_blends_csv
from os import listdir
import networkx as nx

def saldo_obj(filename):
    saldo = defaultdict(int)
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.split('\t')
            pos = line[-2].upper()
            lemma_id = line[0]
            lemma = line[0].split('..')[0].lower()
            mother = line[1]
            father = line[2]
            saldo[lemma] = (pos, father, mother, lemma_id)
    return saldo

# def construct_network(saldo):
#     G = nx.DiGraph()
#     for k, (_, m, f, li) in saldo.items():
#         if m not in G.nodes:
#             G.add_node(m)
#         if f not in G.nodes:
#             G.add_node(m)
#         if li not in G.nodes:
#             G.add_node(li) 
#         if k not in G.nodes:
#             G.add('_' + k)
    
#         if G.has_edge(li, k):
#             G[k][li]['weight'] += 1
#         else:
#             G.add_edge(k, li, weight=1)

#         if G.jas

def get_candidates():
    lexicon = 'saldo'
    corpus = 'news'
    candidate_folder = f'/home/adam/Documents/lexical_blends_project/{lexicon}_blends_candidates_noverlap_1/'
    
    c_set = set()
    for i, filename in enumerate(listdir(candidate_folder)):
        blend = filename.split('_')[0]
        print('### reading blend:', i, blend)
        with open(candidate_folder+filename) as f:
            for ln in f:
                cw1, cw2 = ln.rstrip().split(',')
                c_set.add(cw1)
                c_set.add(cw2)
    return c_set

def nst_obj(filename):
    nst = defaultdict(int)
    with open(filename, encoding='iso-8859-1') as f:
        for i, line in enumerate(f):
            if line.startswith('!') or line.startswith('-'):
                continue

            line = line.split(';')
            seg = line[0]
            pos = line[1]
            sampa = line[11]

            while '|' in pos:
                pos = pos.split('|')[0]
        
            nst[seg.lower()] = (pos, sampa)
    return nst

if __name__ == '__main__':

    #with open('/home/adam/Documents/lexical_blends_project/data/nst_lex.pickle', '+wb') as f:
    #    nst = nst_obj('/home/adam/data/NST_svensk_leksikon/swe030224NST.pron/swe030224NST.pron')
    #    pickle.dump(nst, f)
    
    #with open('/home/adam/Documents/lexical_blends_project/data/saldo_lex.pickle', '+wb') as f:
    #    saldo = saldo_obj('/home/adam/data/saldo_2.3/saldo20v03.txt')
    #    pickle.dump(saldo, f)

    with open('/home/adam/Documents/lexical_blends_project/data/nst_lex.pickle', 'rb') as f:
        nst = pickle.load(f)
    
    with open('/home/adam/Documents/lexical_blends_project/data/saldo_lex.pickle', 'rb') as f:
        saldo = pickle.load(f)


    c_set = get_candidates()

    print(list(saldo.keys())[:100])
    print(list(nst.keys())[:100])
    n_set = set(nst.keys())
    s_set = set(saldo.keys())

    true = len(c_set.intersection(n_set))/len(c_set)
    print(true)



