from helper_functions import *
from collections import defaultdict
from nltk import ngrams
import gensim as gs
from gensim.models.fasttext import FastText
from toolz import keyfilter
from functools import reduce
import pickle
from numpy.linalg import norm
from candidates import blend_keys
from os import listdir
import csv
from multiprocessing import Pool

def get_split_features(blend, sw1, sw2, sw1_split, sw2_split, w1f, w2f, wsm, csm):
    blend = blend
    
    featureset = defaultdict(float)

    possible_splits = list(blend_splits_min2(sw1, sw2, blend))

    # 2:0-2, 3:3-5, 4:6-8, 5:9-11
    w_scores = True
    if w_scores:
        for i in range(2,6):
            ngs = ngrams(sw1, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
            featureset[f'sw1_{i}-gram_score'] = sum(map(lambda x: ngram_embedding_repr(x, csm), ngs))
            ngs = ngrams(sw2, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
            featureset[f'sw2_{i}-gram_score'] = sum(map(lambda x: ngram_embedding_repr(x, csm), ngs))
            ngs = ngrams(blend, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
            featureset[f'blend_{i}-gram_score'] = sum(map(lambda x: ngram_embedding_repr(x, csm), ngs))

        # lägga till? 
        featureset[f'sw1_wordscore'] = ngram_embedding_repr('<' + sw1 + '>', csm)
        featureset[f'sw2_wordscore'] = ngram_embedding_repr('<' + sw2 + '>', csm)
        featureset[f'blend_wordscore'] = ngram_embedding_repr('<' + blend + '>', csm)
    else:
        for i in range(2,7):
            ngs = ngrams(sw1, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
            sw1c = list(map(lambda x: ngram_embedding_repr(x, csm), ngs))
            ngs = ngrams(sw2, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
            sw2c = list(map(lambda x: ngram_embedding_repr(x, csm), ngs))
            ngs = ngrams(blend, i, pad_left=True, pad_right=True, left_pad_symbol='<', right_pad_symbol='>')
            blendc = list(map(lambda x: ngram_embedding_repr(x, csm), ngs))

            # lägga till? 
            sw1c.append(ngram_embedding_repr('<' + sw1 + '>', csm))
            sw2c.append(ngram_embedding_repr('<' + sw2 + '>', csm))
            blendc.append(ngram_embedding_repr('<' + blend + '>', csm))

            featureset[f'{i}_sw1_sw2_ch_emb_sim'] = cos_sim(sw1c, sw2c)
            featureset[f'{i}_sw1_blend_ch_emb_sim'] = cos_sim(sw1c, blendc)
            featureset[f'{i}_sw2_blend_ch_emb_sim'] = cos_sim(sw2c, blendc)

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

    featureset['sw1_sw2_char_bigramsim'] = bigram_sim(sw1,sw2)
    featureset['sw2_sw1_char_bigramsim'] = bigram_sim(sw2,sw1)
    featureset['lcs_sw1_sw2'] = lcs(sw1,sw2)
    featureset['sw1_sw2_lev_dist']   = lev(sw1, sw2)

    featureset['sw1_graphemes'] = grapheme_counter(sw1)/grapheme_counter(blend)
    featureset['sw2_graphemes'] = grapheme_counter(sw2)/grapheme_counter(blend)
    featureset['sw1_syllables'] = syllable_counter(sw1)/syllable_counter(blend)
    featureset['sw2_syllables'] = syllable_counter(sw2)/syllable_counter(blend)
    featureset['sw1_blend_lev_dist'] = lev(sw1, blend)
    featureset['sw2_blend_lev_dist'] = lev(sw2, blend)
    featureset['sw1_len'] = len(sw1)/len(blend)
    featureset['sw2_len'] = len(sw2)/len(blend)
    featureset['sw1_contrib'] = len(sw1_split)/len(blend)
    featureset['sw2_contrib'] = len(sw2_split)/len(blend)
    featureset['sw1_sw2_removal'] = (len(sw1)+len(sw2))/len(blend)
    featureset['splits'] = len(possible_splits)


    sw1_splitp, sw2_splitp = 'V' if sw1[-1] in vowels else 'C', 'V' if sw2[0] in vowels else 'C'
    featureset['split_point_cv'] = sw1_splitp + sw2_splitp

    featureset['split_point_orthography'] = sw1[-1] + sw2[0]
    
    featureset['sw1_aff_c'] = w1f[0]
    featureset['sw1_N_c'] = w1f[1]
    featureset['sw2_aff_c'] = w2f[0]
    featureset['sw2_N_c'] = w2f[1]


    
    return featureset

def freq_features(w, w_split, freqd, pref=True):
    wsum, affsum = freqd[w], 0
    corpus_sum = sum(freqd.values())
    if pref:
        affsum = sum(keyfilter(lambda x: x.startswith(w_split), freqd).values())
    else:
        affsum = sum(keyfilter(lambda x: x.endswith(w_split), freqd).values())

    w1f = wsum/affsum if affsum > 0 else 0.0
    c1f = wsum/corpus_sum

    return [w1f, c1f]

def extract_sample_features(blend, cw1, cw2, lexicon, corpus, sw1, sw2, freqd):
    w1_prefix_dict = defaultdict(list)
    w2_suffix_dict = defaultdict(list)
    T, F = 0, 0

    features = []

    wg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/word_embeddings/corpora/w2v_newsa_min1'
    wsm = gs.models.Word2Vec.load(wg_path)
    cg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/char_embeddings/full_embeddings_window3_skipgram_negsampling'
    csm = gs.models.Word2Vec.load(cg_path)

    for cw1_prefix, cw2_suffix in blend_splits_min2(cw1, cw2, blend):
        if cw1_prefix not in w1_prefix_dict:
            w1_prefix_dict[cw1_prefix] = freq_features(cw1, cw1_prefix,
                                                        freqd, 
                                                        True)

        if cw2_suffix not in w2_suffix_dict:
            w2_suffix_dict[cw2_suffix] = freq_features(cw2, cw2_suffix, 
                                                        freqd, 
                                                        False)

        fset = get_split_features(blend, cw1, cw2, 
                                  cw1_prefix, cw2_suffix,
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
        fset['CW1_split'] = cw1_prefix
        fset['CW2_split'] = cw2_suffix
        
        features.append([fset, label])

    return features

def write_features_to_csv():
    lexicon = 'saldo'
    corpus = 'news'
    gold_blends = blend_keys()

    csvf = open(f'{lexicon}_features_overlap_blends_min1.csv', '+w', newline='')
    csvw = csv.writer(csvf, delimiter=',')

    T, F = 0, 0

    dataf = f'/home/adam/Documents/lexical_blends_project/lexicon_wordlists/{lexicon}_{corpus}_wordlist_f.pickle'

    with open(dataf, 'rb') as f:
        freqd = pickle.load(f)

    candidate_folder = f'/home/adam/Documents/lexical_blends_project/{lexicon}_blend_candidates_1/'

    for i, filename in enumerate(listdir(candidate_folder)):
        blend = filename.split('_')[0]
        print('#', i ,'reading', blend, 'from', candidate_folder+filename)
        with open(candidate_folder+filename) as f:

            for ln in f:
                cw1, cw2 = ln.rstrip().split(',')
                sw1, sw2 = gold_blends[blend]

                #print('### blend:', blend, 'gold:', (sw1, sw2), 'sample:', (cw1, cw2))
                feature_set = extract_sample_features(blend, cw1, cw2, lexicon, corpus, sw1, sw2, freqd)
                for features, label in feature_set:
                    if not features:
                        continue
                    if label:
                        T += 1
                    else:
                        F += 1

                    entry = list(map(lambda x: str(x), features.values()))
                    csvw.writerow(entry)
        print(blend, T, F)

    csvf.close()

def multip_write_features_to_csv():
    lexicon = 'saldo'
    corpus = 'news'
    gold_blends = blend_keys()

    csvf = open('{0}_features_overlap_split_blends_charsim_280718.csv'.format(lexicon), '+w', newline='')
    csvw = csv.writer(csvf, delimiter=',')

    T, F = 0, 0

    dataf = f'/home/adam/Documents/lexical_blends_project/lexicon_wordlists/{lexicon}_{corpus}_wordlist_f.pickle'

    with open(dataf, 'rb') as f:
        freqd = pickle.load(f)

    #candidate_folder = f'/home/adam/Documents/lexical_blends_project/{lexicon}_blend_candidates_1/'
    candidate_folder = '/home/adam/Documents/lexical_blends_project/saldo_blend_candidates_1/'
    
    cand_set = []

    for i, filename in enumerate(listdir(candidate_folder)):
        blend = filename.split('_')[0]
        #print('#', i ,'reading', blend, 'from', candidate_folder+filename)
        with open(candidate_folder+filename) as f:
            for ln in f:
                cw1, cw2 = ln.rstrip().split(',')
                if blend in [cw1, cw2]:
                    continue
                sw1, sw2 = gold_blends[blend]
                cand_set.append((blend, cw1, cw2, lexicon, corpus, sw1, sw2, freqd))

    for cand_chunk in chunks(cand_set, 10):
        with Pool(3) as p:
            entires = p.starmap(extract_sample_features, cand_chunk)
            print('# writing entries')
            for entry in entires:
                for e in entry:
                    csvw.writerow(list(map(lambda x: str(x), e[0].values())))

    csvf.close()

if __name__ == '__main__':
    #multip_write_features_to_csv()

    print('# READING WORD2VEC')
    wg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/word_embeddings/corpora/w2v_newsa_min1'
    wsm = gs.models.Word2Vec.load(wg_path)
    print('# READING FASTTEXT')
    cg_path = '/home/adam/Documents/lexical_blends_project/embeddings/saldo_embeddings_window5_skipgram_negsampling_fasttext'
    csm = FastText.load(cg_path)
    print(csm.similarity('motor', 'hotell'))
    # blend = 'motell'
    # sw1, sw2 = 'motor', 'hotell'
    # sw1_split = 'mot'
    # sw2_split = 'ell'
    # w1f = [100, 100]
    # w2f = [100, 100]
    
    # g = get_split_features(blend, sw1, sw2, sw1_split, sw2_split, w1f, w2f, wsm, csm)

    # features = g.keys()
    # for i, (k, v) in enumerate(g.items()):
    #     print(i, k, v)

    # blend = 'motell'
    # sw1, sw2 = 'motor', 'hotell'
    # sw1_split = 'mot'
    # sw2_split = 'ell'
    # w1f = [100, 100]
    # w2f = [100, 100]
    
    # g = get_split_features(blend, sw1, sw2, sw1_split, sw2_split, w1f, w2f, wsm, csm)