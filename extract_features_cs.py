from helper_functions import *
from collections import defaultdict
from nltk import ngrams
import gensim as gs
from toolz import keyfilter
from functools import reduce
import pickle
from numpy.linalg import norm
from candidates import blend_keys
from os import listdir
import csv

def get_features(blend, sw1, sw2, w1f, w2f, wsm, csm):
    blend = blend
    
    featureset = defaultdict(float)

    possible_splits = list(blend_splits_min2(sw1, sw2, blend))

    for i in range(2,6):
        ngs = ngrams(sw1, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
        featureset[f'sw1_{i}-gram_score'] = sum(map(lambda x: ngram_embedding_repr(x, csm), ngs))
        ngs = ngrams(sw2, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
        featureset[f'sw2_{i}-gram_score'] = sum(map(lambda x: ngram_embedding_repr(x, csm), ngs))
        ngs = ngrams(blend, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
        featureset[f'blend_{i}-gram_score'] = sum(map(lambda x: ngram_embedding_repr(x, csm), ngs))

    featureset['splits'] = len(possible_splits)
        
    ### structural similarity
    featureset['sw1_sw2_char_bigramsim'] = bigram_sim(sw1,sw2)
    featureset['sw2_sw1_char_bigramsim'] = bigram_sim(sw2,sw1)
    featureset['lcs_sw1_sw2'] = lcs(sw1,sw2)

    featureset['sw1_blend_lev_dist'] = lev(sw1, blend)
    featureset['sw2_blend_lev_dist'] = lev(sw2, blend)
    featureset['sw1_sw2_lev_dist']   = lev(sw1, sw2)

    featureset['sw1_graphemes'] = grapheme_counter(sw1)/grapheme_counter(blend)
    featureset['sw2_graphemes'] = grapheme_counter(sw2)/grapheme_counter(blend)

    featureset['sw1_len'] = len(sw1)/len(blend)
    featureset['sw2_len'] = len(sw2)/len(blend)

    featureset['sw1_syllables'] = syllable_counter(sw1)/syllable_counter(blend)
    featureset['sw2_syllables'] = syllable_counter(sw2)/syllable_counter(blend)

    sw1_contribution, sw2_contribution = 0, 0
    for p, s in possible_splits:
        sw1_contribution += len(p)
        sw2_contribution += len(s)
    featureset['sw1_contrib'] = sw1_contribution/len(possible_splits)
    featureset['sw2_contrib'] = sw2_contribution/len(possible_splits)

    # TODO >>> duplicate with 'splits'
    #featureset['sw1_sw2_overlap'] = 0 if len(possible_splits) <= 1 else len(possible_splits)
    featureset['sw1_sw2_removal'] = (len(sw1)+len(sw2))/len(blend)
    
    featureset['sw1_aff_c'] = w1f[0]/len(possible_splits)
    featureset['sw1_N_c'] = w1f[1]/len(possible_splits)
    
    featureset['sw2_aff_c'] = w2f[0]/len(possible_splits)
    featureset['sw2_N_c'] = w2f[1]/len(possible_splits)
        
    if blend in wsm and sw1 in wsm:
        featureset['sw1_LB_sim'] = wsm.similarity(blend, sw1)
    else:
        featureset['sw1_LB_sim'] = 0

    if blend in wsm and sw2 in wsm:
        featureset['sw2_LB_sim'] = wsm.similarity(blend, sw2)
    else:
        featureset['sw2_LB_sim'] = 0
        
    if sw1 in wsm and sw2 in wsm:
        featureset['sw1_sw2_sim'] = wsm.similarity(sw1, sw2)
    else:
        featureset['sw1_sw2_sim'] = 0
    
    return featureset

def freq_features(w, w_split, lexicon, corpora, pref=True):
    dataf = f'/home/adam/Documents/lexical_blends_project/lexicon_wordlists/{lexicon}_{corpora}_wordlist_f.pickle'

    with open(dataf, 'rb') as f:
        freqd = pickle.load(f)

    wsum, affsum = freqd[w], 0
    corpus_sum = sum(freqd.values())
    if pref:
        affsum = sum(keyfilter(lambda x: x.startswith(w_split), freqd).values())
    else:
        affsum = sum(keyfilter(lambda x: x.endswith(w_split), freqd).values())

    w1f = wsum/affsum if affsum > 0 else 0.0
    c1f = wsum/corpus_sum

    return [w1f, c1f]

def extract_sample_features(blend, cw1, cw2, lexicon, corpus, sw1, sw2):
    w1_prefix_dict = defaultdict(list)
    w2_suffix_dict = defaultdict(list)
    T, F = 0, 0

    features = []

    wg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/word_embeddings/corpora/w2v_newsa_min1'
    wsm = gs.models.Word2Vec.load(wg_path)
    cg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/char_embeddings/full_embeddings_window3_skipgram_negsampling'
    csm = gs.models.Word2Vec.load(cg_path)

    cw_splits = list(blend_splits_min2(cw1, cw2, blend))

    for cw1_prefix, cw2_suffix in cw_splits:
        if cw1_prefix not in w1_prefix_dict:
            w1_prefix_dict[cw1_prefix] += freq_features(cw1, cw1_prefix,
                                                        lexicon, corpus, 
                                                        True)

        if cw2_suffix not in w2_suffix_dict:
            w2_suffix_dict[cw2_suffix] += freq_features(cw2, cw2_suffix, 
                                                        lexicon, corpus, 
                                                        False)
    

    fset = get_features(blend, cw1, cw2,
                        w1_prefix_dict[cw1_prefix],
                        w2_suffix_dict[cw2_suffix],
                        wsm, csm)
        
    # determine label
    if (cw1,cw2) == (sw1,sw2):
        T += 1
        label = True
        fset['LABEL'] = True
    else:
        F += 1
        label = False
        fset['LABEL'] = False
        
    fset['BLEND'] = blend
    fset['CW1'] = cw1
    fset['CW2'] = cw2
    fset['CW1_split'] = list(filter(lambda x: x[0], cw_splits))
    fset['CW2_split'] = list(filter(lambda x: x[1], cw_splits))
    
    #features.append([fset, label])

    return fset, label

def write_features_to_csv():
    lexicon = 'saldo'
    corpus = 'news'
    gold_blends = blend_keys()

    csvf = open(f'{lexicon}_features_noverlap_blends_min1_samplewords.csv', '+w', newline='')
    csvw = csv.writer(csvf, delimiter=',')

    T, F = 0, 0

    candidate_folder = f'/home/adam/Documents/lexical_blends_project/{lexicon}_blends_candidates_noverlap_1/'
    
    for i, filename in enumerate(listdir(candidate_folder)):
        blend = filename.split('_')[0]
        print('### reading blend:', i, blend)
        with open(candidate_folder+filename) as f:
            for ln in f:
                cw1, cw2 = ln.rstrip().split(',')
                sw1, sw2 = gold_blends[blend]
                
                feature_set, label = extract_sample_features(blend, cw1, cw2, lexicon, corpus, sw1, sw2)
                entry = list(map(lambda x: str(x), feature_set.values()))

                if label == True:
                    T += 1
                else:
                    F += 1


                csvw.writerow(entry)
        print(blend, T, F)

    csvf.close()
            

if __name__ == '__main__':
    write_features_to_csv()
    #f, l = extract_sample_features('motell', 'motor', 'hotell', 'saldo', 'news', 'motor', 'hotell')
    #for k, v in f.items():
    #    print(k, v)