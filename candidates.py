from toolz import keymap, keyfilter, valfilter
from itertools import groupby, product, chain
from collections import defaultdict, Counter
import pickle
from os import listdir
from helper_functions import load_blends_from_csv, blend_splits_min2
import csv

def get_words(start, end, lexicon):
    startl = set(filter(lambda x: x.startswith(start), lexicon))
    endl = set(filter(lambda x: x.endswith(end), lexicon))
    return set(product(startl, endl))

def load_lexicon(lpath):
    with open(lpath, 'rb') as f:
        return filter_lexicon(pickle.load(f))
        #return pickle.load(f)

def filter_lexicon(lexicon):
    return valfilter(lambda x: x != 0, lexicon)

def read_blends(bpath):
    blends_dataset = set()
    for blend, w1, w2 in load_blends_from_csv():
        blend, w1, w2 = blend[:-3], w1[:-3], w2[:-3]
        blends_dataset.add((blend, w1, w2))
    return blends_dataset

# ONE OR MORE SPLITS
def filter_overlapping_blends(blends):
    return set(filter(lambda x: len(list(blend_splits_min2(x[1], x[2], x[0]))) >= 1, blends))

def extract_candidates_from_lexicon(lexicon, blend, min_startend=2, overlap=0):
    lexicon = set(filter(lambda x: '_' not in x, lexicon.keys()))
    prefix = set(filter(lambda x: x.startswith(blend[:min_startend]), lexicon))
    suffix = set(filter(lambda x: x.endswith(blend[-min_startend:]), lexicon))

    if overlap == 0:
        candidate_set = set(filter(
                                lambda x: len(list(blend_splits_min2(x[0], x[1], blend))) >= 2, 
                                product(prefix, suffix)))
    elif overlap == 1:
        candidate_set = set(filter(
                                lambda x: len(list(blend_splits_min2(x[0], x[1], blend))) >= 1, 
                                product(prefix, suffix))) 
    else:
        candidate_set = set(filter(
                                lambda x: len(list(blend_splits_min2(x[0], x[1], blend))) == 1 
                                                    and (len(x[0]) + len(x[1]))/len(blend) <= 3.5, 
                                product(prefix, suffix)))
    return candidate_set

def save_candidate_set_to_file(filename, candidate_set):
    with open(filename, '+w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for cw1, cw2 in candidate_set:
            csv_writer.writerow([cw1, cw2])
    return 0 

def blend_keys():
    blend_dict = defaultdict(tuple)
    blends = read_blends('/home/adam/Documents/lexical_blends_project/blend_wordlists/all_blends.csv')
    for a, b, c in blends:
        blend_dict[a] = (b, c)
    return blend_dict


def save_candidates_overlap_blends(lexicon_name):
    lexicon_path = f'/home/adam/Documents/lexical_blends_project/lexicon_wordlists/{lexicon_name}_news_wordlist_f.pickle'
    bpath = '/home/adam/Documents/lexical_blends_project/blend_wordlists/all_blends.csv'
    lexicon = load_lexicon(lexicon_path)
    blends = read_blends(bpath)
    overlap_blends = filter_overlapping_blends(blends)
    #noverlap_blends = blends.difference(overlap_blends)

    blend_cs_folder = f'/home/adam/Documents/lexical_blends_project/{lexicon_name}_blend_candidates_overlap_1'

    for blend, sw1, sw2 in overlap_blends:
        blend_cs = extract_candidates_from_lexicon(lexicon, blend, 2, 0)

        if (sw1, sw2) in blend_cs:
            print('# writing', blend, 'candidate set size =', len(blend_cs))
            filename = f'{blend_cs_folder}/{blend}_candidates_overlap.csv'
            save_candidate_set_to_file(filename, blend_cs)

def save_candidates_noverlap_blends(lexicon_name):
    lexicon_path = f'/home/adam/Documents/lexical_blends_project/lexicon_wordlists/{lexicon_name}_news_wordlist_f.pickle'
    bpath = '/home/adam/Documents/lexical_blends_project/blend_wordlists/all_blends.csv'
    lexicon = load_lexicon(lexicon_path)
    blends = read_blends(bpath)
    overlap_blends = filter_overlapping_blends(blends)
    noverlap_blends = blends.difference(overlap_blends)

    blend_cs_folder = f'/home/adam/Documents/lexical_blends_project/{lexicon_name}_blends_candidates_noverlap_1'
    #blend_cs_folder = '/home/adam/Documents/lexical_blends_project/blend_candidates'

    for blend, sw1, sw2 in noverlap_blends:
        blend_cs = extract_candidates_from_lexicon(lexicon, blend, 2, 2)
        #print((sw1, sw2) in blend_cs, len(blend_cs))
        if (sw1, sw2) in blend_cs:
            print('# writing', blend, 'candidate set size =', len(blend_cs))
            filename = f'{blend_cs_folder}/{blend}_candidates_noverlap.csv'
            save_candidate_set_to_file(filename, blend_cs)


if __name__ == '__main__':
    save_candidates_noverlap_blends('saldo')


