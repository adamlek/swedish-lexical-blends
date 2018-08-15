from helper_functions import *
import pickle
from collections import Counter

def lengths(blends):
    sw1, sw2 = [], []
    for _, w1, w2 in blends:
        sw1.append(len(w1))
        sw2.append(len(w2))
    
    print(sw1)
    print(sw2)

def grapheme_test(blends):
    sw1, sw2 = [], []
    for _, w1, w2 in blends:
        sw1.append(grapheme_counter(w1))
        sw2.append(grapheme_counter(w2))
    
    print(sw1)
    print(sw2)

def syll_test(blends):
    sw1, sw2 = [], []
    for _, w1, w2 in blends:
        sw1.append(syllable_counter(w1))
        sw2.append(syllable_counter(w2))
    
    print(sw1)
    print(sw2)

def freq_test(blends):
    
    with open('/home/adam/Documents/lexical_blends_project/lexicon_wordlists/saldo_news_wordlist_f.pickle', 'rb') as f:
        freqd = pickle.load(f)

    sw1, sw2 = [], []
    for _, w1, w2 in blends:
        if w1 in freqd and w2 in freqd:
            sw1.append(freqd[w1])
            sw2.append(freqd[w2])
    
    print(sw1)
    print(sw2)
    print(len(sw1))

def contribution(blends):
    short, long = [], []
    for blend, sw1, sw2 in blends:
        if len(sw1) > len(sw2):
            longw, shortw = sw1, sw2
        else:
            longw, shortw = sw2, sw1

        sw1_c, sw2_c = 0, 0
        splits = list(blend_splits_min2(sw1, sw2, blend))
        if not splits:
            continue
        for s in splits:
            sw1_c += len(s[0])
            sw2_c += len(s[1])
        
        
        if longw == sw1:
            long.append(sw1_c/len(splits))
            short.append(sw2_c/len(splits))
        else:
            long.append(sw2_c/len(splits))
            short.append(sw1_c/len(splits))   

    print(short)
    print(long)        


if __name__ == '__main__':
    blends = [(x.word[:-3], x.w1[:-3], x.w2[:-3]) for x in load_blends_from_csv()]

    b = [x[0] for x in blends]
    sw1 = [x[1] for x in blends]
    sw2 = [x[2] for x in blends]

    a = Counter(sw1)
    b = Counter(sw2)

    print(a)
    print(b)

    #contribution(blends)
