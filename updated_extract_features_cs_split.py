from helper_functions import *
from collections import defaultdict
from nltk import ngrams
import gensim as gs
from gensim.models.fasttext import FastText
from operator import mul
from toolz import keyfilter
from functools import reduce
import pickle
from numpy.linalg import norm
from candidates import blend_keys
from os import listdir
import csv
from multiprocessing import Pool
import epitran
from pyphonetics.distance_metrics import levenshtein_distance as ipa_lev

def ngram_vector(ngram, csm):
    v = csm[ngram[0]]
    for l in ngram[1:]:
        #print(v)
        #print(csm[l])
        v = v*csm[l]
    return v

def get_split_features(blend, sw1, sw2, sw1_split, sw2_split, w1f, w2f, wsm, csm, epit):
    blend = blend
    
    featureset = defaultdict(float)

    possible_splits = list(blend_splits_min2(sw1, sw2, blend))

    if sw1 in csm:
        featureset['sw1_charemb_score'] = sum(csm[sw1])
    else:
        featureset['sw1_charemb_score'] = 0

    if sw2 in csm:
        featureset['sw2_charemb_score'] = sum(csm[sw2])
    else:
        featureset['sw2_charemb_score'] = 0

    if blend in csm:
        featureset['blend_charemb_score'] = sum(csm[blend])
    else:
        featureset['blend_charemb_score'] = 0

    # 2:0-2, 3:3-5, 4:6-8, 5:9-11
    if sw1 in csm and sw2 in csm:
        featureset['sw1_sw2_charemb_sim'] = csm.similarity(sw1, sw2)
    else:
        featureset['sw1_sw2_charemb_sim'] = 0
    
    if sw1 in csm and blend in csm:
        featureset['sw1_blend_charemb_sim'] = csm.similarity(sw1, blend)
    else:
        featureset['sw1_blend_charemb_sim'] = 0
    
    if sw2 in csm and blend in csm:
        featureset['sw2_blend_charemb_sim'] = csm.similarity(sw2, blend)
    else:
        featureset['sw2_blend_charemb_sim'] = 0
    
    # WSM
    if sw1 in wsm:
        featureset['sw1_wordemb_score'] = sum(wsm[sw1])
    else:
        featureset['sw1_wordemb_score'] = 0
    
    if sw2 in wsm:
        featureset['sw2_wordemb_score'] = sum(wsm[sw2])
    else:
        featureset['sw2_wordemb_score'] = 0

    if blend in wsm:
        featureset['blend_wordemb_score'] = sum(wsm[sw1])
    else:
        featureset['blend_wordemb_score'] = 0

    if blend in wsm and sw1 in wsm:
        featureset['sw1_blend_wordemb_sim'] = wsm.similarity(blend, sw1)
    else:
        featureset['sw1_blend_wordemb_sim'] = 0

    if blend in wsm and sw2 in wsm:
        featureset['sw2_blend_wordemb_sim'] = wsm.similarity(blend, sw2)
    else:
        featureset['sw2_blend_wordemb_sim'] = 0
        
    if sw1 in wsm and sw2 in wsm:
        featureset['sw1_sw2_wordemb_sim'] = wsm.similarity(sw1, sw2)
    else:
        featureset['sw1_sw2_wordemb_sim'] = 0

    featureset['splits'] = len(possible_splits)
        
    featureset['sw1_sw2_char_bigramsim'] = bigram_sim(sw1,sw2)
    featureset['sw2_sw1_char_bigramsim'] = bigram_sim(sw2,sw1)
    featureset['sw1_sw2_char_trigramsim'] = trigram_sim(sw1,sw2)
    featureset['sw2_sw1_char_trigramsim'] = trigram_sim(sw2,sw1)
    featureset['lcs_sw1_sw2'] = lcs(sw1,sw2)

    sw1_ipa = epit.transliterate(sw1)
    sw2_ipa = epit.transliterate(sw2)
    blend_ipa = epit.transliterate(blend)
    featureset['sw1_blend_IPA_lev_dist'] = ipa_lev(sw1_ipa, blend_ipa)
    featureset['sw2_blend_IPA_lev_dist'] = ipa_lev(sw2_ipa, blend_ipa)
    featureset['sw1_sw2_IPA_lev_dist']   = ipa_lev(sw1_ipa, sw2_ipa)

    featureset['sw1_blend_lev_dist'] = lev(sw1, blend)
    featureset['sw2_blend_lev_dist'] = lev(sw2, blend)
    featureset['sw1_sw2_lev_dist']   = lev(sw1, sw2)

    featureset['sw1_graphemes'] = grapheme_counter(sw1)/grapheme_counter(blend)
    featureset['sw2_graphemes'] = grapheme_counter(sw2)/grapheme_counter(blend)
    
    featureset['sw1_syllables'] = syllable_counter(sw1)/syllable_counter(blend)
    featureset['sw2_syllables'] = syllable_counter(sw2)/syllable_counter(blend)

    featureset['sw1_len'] = len(sw1)/len(blend)
    featureset['sw2_len'] = len(sw2)/len(blend)
    featureset['sw1_contrib'] = len(sw1_split)/len(blend)
    featureset['sw2_contrib'] = len(sw2_split)/len(blend)

    featureset['sw1_sw2_removal'] = (len(sw1)+len(sw2))/len(blend)
    
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

def extract_sample_features(blend, cw1, cw2, lexicon, corpus, sw1, sw2, freqd, wsm, csm, epit):
    w1_prefix_dict = defaultdict(list)
    w2_suffix_dict = defaultdict(list)
    T, F = 0, 0

    features = []

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
                                  wsm, csm, epit)
        
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

    wg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/word_embeddings/corpora/w2v_newsa_min1'
    wsm = gs.models.Word2Vec.load(wg_path)
    cg_path = '/home/adam/Documents/lexical_blends_project/embeddings/cc.sv.300.bin'
    csm = FastText.load_fasttext_format(cg_path)
    #cg_path = '/home/adam/Documents/lexical_blends_project/embeddings/saldo_embeddings_window5_skipgram_negsampling_fasttext'
    #csm = FastText.load(cg_path)
    epit = epitran.Epitran('swe-Latn')

    col_names = ['sw1_charemb_score', 'sw2_charemb_score', 'blend_charemb_score',
                 'sw1_sw2_charemb_sim', 'sw1_blend_charemb_sim', 'sw2_blend_charemb_sim', 
                 'sw1_wordemb_score', 'sw2_wordemb_score', 'blend_wordemb_score',
                 'sw1_blend_wordemb_sim', 'sw2_blend_wordemb_sim', 'sw1_sw2_wordemb_sim', 
                 'splits', 
                 'sw1_sw2_char_bigramsim', 'sw2_sw1_char_bigramsim', 'sw1_sw2_char_trigramsim', 'sw2_sw1_char_trigramsim', 
                 'lcs_sw1_sw2', 
                 'sw1_blend_IPA_lev_dist', 'sw2_blend_IPA_lev_dist', 'sw1_sw2_IPA_lev_dist', 
                 'sw1_blend_lev_dist', 'sw2_blend_lev_dist', 'sw1_sw2_lev_dist', 
                 'sw1_graphemes', 'sw2_graphemes', 
                 'sw1_syllables', 'sw2_syllables', 
                 'sw1_len', 'sw2_len', 
                 'sw1_contrib', 'sw2_contrib', 'sw1_sw2_removal', 
                 'sw1_aff_c', 'sw1_N_c', 'sw2_aff_c', 'sw2_N_c', 
                 'LABEL', 'BLEND', 'CW1', 'CW2', 'CW1_split', 'CW2_split']

    csvf = open('{0}_features_split_OVERLAP_fasttext_060818.csv'.format(lexicon), '+w', newline='')
    csvw = csv.DictWriter(csvf, delimiter=',', fieldnames=col_names)

    T, F = 0, 0

    dataf = f'/home/adam/Documents/lexical_blends_project/lexicon_wordlists/{lexicon}_{corpus}_wordlist_f.pickle'

    with open(dataf, 'rb') as f:
        freqd = pickle.load(f)

    # overlap
    candidate_folder = '/home/adam/Documents/lexical_blends_project/saldo_blend_candidates_1/'
    # noverlap
    #candidate_folder = '/home/adam/Documents/lexical_blends_project/saldo_blends_candidates_noverlap_1/'

    for i, filename in enumerate(listdir(candidate_folder)):
        blend = filename.split('_')[0]
        print('#', i ,'reading', blend)
        with open(candidate_folder+filename) as f:

            for ln in f:
                cw1, cw2 = ln.rstrip().split(',')
                sw1, sw2 = gold_blends[blend]

                #print('### blend:', blend, 'gold:', (sw1, sw2), 'sample:', (cw1, cw2))
                feature_set = extract_sample_features(blend, cw1, cw2, lexicon, corpus, sw1, sw2, freqd, wsm, csm, epit)
                for features, label in feature_set:
                    #entry = list(map(lambda x: str(x), features.values()))
                    csvw.writerow(features)

    csvf.close()

def multip_write_features_to_csv():
    lexicon = 'saldo'
    corpus = 'news'
    gold_blends = blend_keys()

    wg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/word_embeddings/corpora/w2v_newsa_min1'
    wsm = gs.models.Word2Vec.load(wg_path)
    cg_path = '/home/adam/Documents/lexical_blends_project/embeddings/saldo_embeddings_window5_skipgram_negsampling_fasttext'
    csm = gs.models.Word2Vec.load(cg_path)
    epit = epitran.Epitran('swe-Latn')

    csvf = open('{0}_features_overlap_split_020818.csv'.format(lexicon), '+w', newline='')
    csvw = csv.writer(csvf, delimiter=',')

    T, F = 0, 0

    dataf = f'/home/adam/Documents/lexical_blends_project/lexicon_wordlists/{lexicon}_{corpus}_wordlist_f.pickle'

    with open(dataf, 'rb') as f:
        freqd = pickle.load(f)

    # overlap
    candidate_folder = '/home/adam/Documents/lexical_blends_project/saldo_blend_candidates_1/'
    # noverlap
    #candidate_folder = '/home/adam/Documents/lexical_blends_project/saldo_blends_candidates_noverlap_1/'
    
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
                cand_set.append((blend, cw1, cw2, lexicon, corpus, sw1, sw2, freqd, csm, wsm, epit))

    for cand_chunk in chunks(cand_set, 10):
        with Pool(3) as p:
            entires = p.starmap(extract_sample_features, cand_chunk)
            print('# writing entries')
            for entry in entires:
                for e in entry:
                    csvw.writerow(list(map(lambda x: str(x), e[0].values())))

    csvf.close()

if __name__ == '__main__':
    write_features_to_csv()

    # print('# READING word2vec')
    # wg_path = '/home/adam/Documents/Magisteruppsats_VT18/ddata/word_embeddings/corpora/w2v_newsa_min1'
    # wsm = gs.models.Word2Vec.load(wg_path)
    # print('# READING FASTTEXT')
    # cg_path = '/home/adam/Documents/lexical_blends_project/embeddings/saldo_embeddings_window5_skipgram_negsampling_fasttext'
    # csm = FastText.load(cg_path)
    # blend = 'motell'
    # sw1, sw2 = 'motor', 'hotell'
    # sw1_split = 'mot'
    # sw2_split = 'ell'
    # w1f = [100, 100]
    # w2f = [100, 100]
    # epit = epitran.Epitran('swe-Latn')
    
    # g = get_split_features(blend, sw1, sw2, sw1_split, sw2_split, w1f, w2f, wsm, csm, epit)

    # print(g.keys())

    # blend = 'motell'
    # sw1, sw2 = 'motor', 'hotell'
    # sw1_split = 'mot'
    # sw2_split = 'ell'
    # w1f = [100, 100]
    # w2f = [100, 100]
    # epit = epitran.Epitran('swe-Latn')
    
    # g = get_split_features(blend, sw1, sw2, sw1_split, sw2_split, w1f, w2f, wsm, csm, epit)

    # for i, (k, v) in enumerate(g.items()):
    #     print(i, k, v)