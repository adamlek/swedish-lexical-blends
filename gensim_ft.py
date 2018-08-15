import gensim as gs
from gensim.models.fasttext import FastText

class SentsFromCorpus():
    def __init__(self, path):
        self.path = path
        
    def __iter__(self):
        with open(self.path) as f:
            for ln in f:
                if ln == '\n':
                    continue
                yield [x.split('_')[0].lower() for x in ln.split('\t')]

def create_model(dpath, model_name, linear_k):
    # lpath = .txt containing one word per line
    #chars = CharsFromLexicon(dpath)
    sents = SentsFromCorpus(dpath)

    print('# creating model')
    model = gs.models.FastText(sents, min_count=1, window=linear_k, sg=1, negative=5, word_ngrams=1, min_n=2, max_n=6, workers=4)

    print('# saving model')
    model.save(model_name)


if __name__ == '__main__':
    mainl = '/home/adam/Documents/Magisteruppsats_VT18/ddata'
    linear_k = 5
    lexicon = 'saldo'
    lpath = f'{mainl}/wordlists/{lexicon}_words.txt'
    model_name = f'{mainl}/char_embeddings/{lexicon}_embeddings_window{linear_k}_skipgram_negsampling_fasttext'
    cpath = f'/home/adam/data/news/sentence_segmented/newscorpus.txt'

    create_model(cpath, model_name, linear_k)