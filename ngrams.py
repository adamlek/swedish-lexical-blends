from collections import defaultdict
from nltk import ngrams
import pickle

class SentsFromCorpus():
    def __init__(self, path):
        self.path = path
        
    def __iter__(self):
        with open(self.path) as f:
            for ln in f:
                if ln == '\n':
                    continue
                yield [x.split('_')[0].lower() for x in ln.split('\t')]

class SentsFromCorpusPP():
    def __init__(self, path):
        self.path = path
        
    def __iter__(self):
        with open(self.path) as f:
            for ln in f:
                if ln == '\n':
                    continue
                if 'PP' in ln:
                    yield [x for x in ln.split('\t')]

def news_ngrams(cpath):
    corpus_sentences = SentsFromCorpusPP(cpath)
    ngram_dict = defaultdict(int)

    folder_path = '/home/adam/Documents/lexical_blends_project/lexicon_wordlists/'
    with open(folder_path + 'saldo_news_wordlist_f.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    for i, sentence in enumerate(corpus_sentences):
        for ng in ngrams(sentence, 3):
            if 'PP' in ng[1]:
                #print(ng)
                w1 = ng[0].split('_')[0]
                w2 = ng[2].split('_')[0]
                if w1 in lexicon and w2 in lexicon:
                    ngram_dict[(w1,w2)] += 1
    

    for k, v in ngram_dict.items():
        ngram_dict[k] = ngram_dict[k]/lexicon[k[1]]

    # (w_-1, w_0)
    with open('3-gramsPP_saldo.pickle', '+wb') as o:
        pickle.dump(ngram_dict, o)
    


if __name__ == '__main__':
    news_ngrams('/home/adam/data/news/sentence_segmented/newscorpus.txt')