from toolz import sliding_window
from collections import namedtuple, defaultdict
from itertools import accumulate
from functools import reduce
import numpy as np
from numpy.linalg import norm
import csv
from pydata import sampa_translations

def cos_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

Word = namedtuple('Word', 'word w1 w2')
vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'å', 'ä', 'ö']
alphabet = 'abcdefghijklmnopqrstuvwåäö'

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
                'sw1_aff_c', 'sw1_N_c', 'sw2_aff_c', 'sw2_N_c']

def feature_indices():
    fdict = defaultdict(str)
    for i, f in enumerate(col_names):
        fdict[i] = f
    return fdict

def ngram_embedding_repr(ngram, model):
    #return sum(reduce(lambda x, y: x*y, [model[x] for x in ngram]))
    return sum(reduce(lambda x, y: x*y, map(lambda x: model[x], ngram)))

def format_lemma(lemma):
    while '|' in lemma:
        lemma = lemma.split('|')[0]
    return lemma

def fix_key(s):
    return s.split('/')[-1]

def word_repr(w):
    return np.array([1 if x in vowels else 0 for x in w])

def one_hot_char(c):
    return np.array([1 if x == c else 0 for x in alphabet])

def swedish_letters(word):
    return len([x for x in word if x in alphabet]) == len(word)

def detect_compounds(word, lexicon):
    for subst in accumulate(word):
        pass

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def phoneme_representation(word):
    pass

def grapheme_counter(representation):
    translation = []
    c = 0
    for t in sliding_window(2, representation):
        if t[0]==t[1]:
            c += 1
        elif ''.join(t) in sampa_translations.values():
            c += 1
        elif t[0] in vowels and t[1] in vowels:
            c += 1
        else:
            translation.append(t[0]) 
    translation.append(t[1])

    return len(representation)-c

def sampa_representation(representation):
    translation = []
    c = 0
    for t in sliding_window(2, representation):
        if t[0]==t[1]:
            c += 1
        elif ''.join(t) in sampa_translations.values():
            c += 1
        elif t[0] in vowels and t[1] in vowels:
            c += 1
        else:
            translation.append(t[0]) 
    translation.append(t[1])

    return len(representation)-c

def syllable_counter(word):
    word = word.lower()
    return sum([1 if x in vowels else 0 for x in word.lower()])

def make_word(wordl):
    return Word(wordl[0], wordl[1], wordl[2])

BlendEntry = namedtuple('BlendEntry', 'blend, sw1, sw2')
WordPos = namedtuple('WordPos', 'word pos')

def obsolete_make_word_pos(bl, w1, w2):
    return BlendEntry(WordPos(bl[0], bl[1]),
                      WordPos(w1[0], w1[1]),
                      WordPos(w2[0], w2[1]))

def obsolete_get_blends_csv():
    pathhh = '/home/adam/Documents/Magisteruppsats_VT18/ddata/wordlists/main_pos-swedish-blends.csv'

    blend_list = []
    with open(pathhh) as f:
        for line in f:
            line = line.split(',')
            if line[6] == 'pref-suff':
                entry = make_word_pos((line[0], line[1]), (line[2], line[3]), (line[4], line[5]))
                blend_list.append(entry)
    
    return blend_list

def get_ngram_file(pathhh):
    with open(pathhh) as f:
        return {x:int(v) for (v,x) in [y.split('\t') for y in f.read().split('\n')]}
    
def make_word_no_pos(bl, w1, w2):
    return Word(bl[0], w1[0], w2[0])
    
def make_word_pos(bl, w1, w2):
    return Word('_'.join(bl),'_'.join(w1),'_'.join(w2))

def get_blends_csv():
    pathhh = '/home/adam/Documents/Magisteruppsats_VT18/ddata/wordlists/main_pos-swedish-blends.csv'
    pathhh = '/home/adam/Documents/Magisteruppsats_VT18/ddata/wordlists/blends_overlappers_withkey.csv'

    blend_list = []
    with open(pathhh) as f:
        for line in f:
            line = line.split(',')
            if line[6] == 'pref-suff':
                entry = make_word_pos([line[0].lower(), line[1]], 
                                      [line[2].lower(), line[3]], 
                                      [line[4].lower(), line[5]])
                blend_list.append(entry)
    return blend_list

def load_blends_from_csv():
    pathhh = '/home/adam/Documents/lexical_blends_project/blend_wordlists/all_blends.csv'

    blend_list = []
    with open(pathhh) as f:
        for line in f:
            line = line.split(',')
            if line[6] == 'pref-suff':
                entry = make_word_pos([line[0].lower(), line[1]], 
                                      [line[2].lower(), line[3]], 
                                      [line[4].lower(), line[5]])
                blend_list.append(entry)
    return blend_list

def get_telescope_wordlist(pathhh):
    with open(pathhh) as f:
        return [make_word([y.lower() for y in x.split() if y.isalpha()])
                for x in f.read().split('\n') if x]

def split_word_min2(word):
    return list(map(lambda x: (word[:x+1], word[x+1:]), range(len(word))))[1:-2]

def split_word_min1(word):
    return list(map(lambda x: (word[:x+1], word[x+1:]), range(len(word))))[:-1]

def lcs(word1, word2):
    for i in reversed(range(2,len(word2))):
        for subs in sliding_window(i, word2):
            if ''.join(subs) in word1:
                return i
    return 0

def bigram_sim(word1, word2):
    w1, w2 = letter_grams(word1, 2), letter_grams(word2, 2)
    if len(w1) > 0 and len(w2) > 0:
        return len(list(filter(lambda x: x in w2, w1)))/len(w1)
    else:
        return 0

def trigram_sim(word1, word2):
    w1, w2 = letter_grams(word1, 3), letter_grams(word2, 3)
    if len(w1) > 0 and len(w2) > 0:
        return len(list(filter(lambda x: x in w2, w1)))/len(w1)
    else:
        return 0

def letter_grams(word, n):
    return list(map(lambda x: x[0]+x[1], sliding_window(n, word)))

def rev_str(str):
    return str[::-1]

def word_split_min2(a):
    return list(accumulate(a))[1:]

def get_splits(a, b, blend):
    pref_splits, suff_splits = split_word_min2(a), split_word_min2(b)
    for pref, _ in pref_splits:
        for _, suff in suff_splits:
            if pref+suff == blend:
                yield (pref, suff)

def blend_splits_min2(a, b, blend):
    pref, suff = word_split_min2(a), list(map(lambda x: x[::-1], word_split_min2(b[::-1])))
    for p in pref:
        for s in suff:
            if p+s == blend:
                yield (p,s)

def get_splits_min1(a, b, blend):
    pref_splits, suff_splits = accumulate(a), [x[::-1] for x in accumulate(b[::-1])]
    for pref in pref_splits:
        for suff in suff_splits:
            if pref+suff == blend:
                yield (pref, suff)

def lev(s, t):
    if s == t: return 0
    elif len(s) == 0: return len(t)
    elif len(t) == 0: return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
            
    return v1[len(t)]

if __name__ == '__main__':
    g = get_ngram_file('/home/adam/Documents/Magisteruppsats_VT18/ddata/ngrams/s_nstp_blogs_trigrams.txt')
