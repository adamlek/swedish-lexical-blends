from helper_functions import get_blends_csv, lev, feature_indices
import numpy as np
from itertools import chain
from collections import defaultdict, Counter
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import FeatureHasher
import sklearn.metrics as metrics
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import ClusterCentroids, TomekLinks, OneSidedSelection, AllKNN
from imblearn.ensemble import BalanceCascade
import pickle
from toolz import keyfilter
import matplotlib.pyplot as plt

def featureset_modification(featureset, remove):
    if isinstance(remove, tuple):
        for feature in remove:
            feature = f'f_{feature}'
            del featureset[feature]
    else:
        del featureset[f'f_{remove}']
    return featureset

def resample_dataset(dataset, labelset):
    d = Counter(labelset)
    over_sampling = SMOTE(k_neighbors=2, n_jobs=4, ratio={0:d[0], 1:d[1]*10})
    #under_sampling = AllKNN(n_jobs=4, n_neighbors=2)
    #combined_sampling = SMOTETomek()
    dataset, labelset = over_sampling.fit_sample(dataset, labelset)
    print('## train resample distribution', Counter(labelset))
    return dataset, labelset

def load_all_data(overlap, remove=False, pos=False, sampa=False):
    
    if overlap:
        ### overlapping data
        current_blend = 'kattikett'
        fpath = '/home/adam/Documents/lexical_blends_project/project/saldo_features_split_OVERLAP_fasttext_060818.csv'
    else:
        ### noverlapping data
        current_blend = 'skypebo'
        fpath = '/home/adam/Documents/lexical_blends_project/project/saldo_features_split_NOVERLAP_fasttext_060818.csv'

    lexicon = 'saldo'

    folder = '/home/adam/Documents/lexical_blends_project/data/'

    # with open('/home/adam/Documents/lexical_blends_project/project/2-grams_saldo.pickle', 'rb') as f:
    #     dict_lexicon = pickle.load(f)
    # with open('/home/adam/Documents/lexical_blends_project/project/3-gramsPP_saldo.pickle', 'rb') as fd:
    #     dict3_lexicon = pickle.load(fd)

    T, F = 0,0
    
    #print('# extracting saldo data for', current_blend)
    with open(fpath) as f:
        featureset = [[]]
        labelset = [[]]
        infoset = [[]]
        for num_b, ln in enumerate(f):
            *features, label, bl, w1, w2, w1_split, w2_split = ln.split(',')

            if num_b == 0:
                current_blend = bl
            else:
                pass

            fset = dict([(f'f_{i}', float(x)) for i, x in enumerate(features)])

            # if (w1, w2) in dict_lexicon:
            #     fset[f'f_{len(fset)}'] = dict_lexicon[(w1, w2)]
            #     #print(w1, w2)
            # else:
            #     fset[f'f_{len(fset)}'] = 0
            
            # if (w2, w1) in dict_lexicon:
            #     fset[f'f_{len(fset)}'] = dict_lexicon[(w2, w1)]
            #     #print(w2, w1)
            # else:
            #     fset[f'f_{len(fset)}'] = 0

            # if (w1, w2) in dict3_lexicon:
            #     fset[f'f_{len(fset)}'] = dict3_lexicon[(w1, w2)]
            #     #print(w1, w2)
            # else:
            #     fset[f'f_{len(fset)}'] = 0
            
            # if (w2, w1) in dict3_lexicon:
            #     fset[f'f_{len(fset)}'] = dict3_lexicon[(w2, w1)]
            #     #print(w2, w1)
            # else:
            #     fset[f'f_{len(fset)}'] = 0

            if remove:
                fset = featureset_modification(fset, remove)
            
            if label == 'False':
                label = False
            else:
                label = True

            if bl == current_blend:
                featureset[-1].append(fset)
                infoset[-1].append((bl, w1, w2, w1_split, w2_split))
                labelset[-1].append(label)
            else:
                current_blend = bl
                featureset.append([fset])
                infoset.append([(bl, w1, w2, w1_split, w2_split)])
                labelset.append([label])

    return featureset, labelset, infoset

def save_data():
    f, l, i = load_all_data(False)


def cross_val(dataset, overlap=False, rem=False, verbose=True, top_n=5, all_s=False):
    
    #dataset = 1

    if dataset == 0:
        splits = [[4, 33, 19, 40, 56, 62, 60, 57, 21],
                  [47, 48, 53, 10, 6, 14, 42, 39, 35],
                  [5, 20, 28, 9, 22, 58, 16, 36, 15],
                  [51, 1, 34, 55, 7, 44, 32, 49, 25],
                  [50, 17, 12, 46, 61, 11, 3, 23, 54],
                  [59, 26, 37, 18, 41, 43, 29, 0, 8]]
    elif dataset == 1:
        splits = [[0, 6, 39, 78, 60, 84, 45, 55, 49, 93, 87],
                  [9, 22, 18, 5, 17, 72, 52, 20, 43, 88, 32],
                  [8, 86, 12, 48, 54, 7, 92, 46, 44, 58, 47],
                  [19, 23, 41, 40, 50, 27, 69, 14, 62, 11, 28],
                  [15, 36, 33, 38, 30, 24, 16, 31, 64, 1],
                  [70, 59, 79, 77, 51, 21, 73, 4, 10, 63],
                  [74, 71, 3, 90, 13, 56, 53, 68, 2, 80],
                  [25, 75, 65, 37, 82, 83, 42, 35, 81, 76],
                  [57, 67, 66, 29, 91, 85, 89, 34, 61, 26, ]]
    #all_s = False
    else:
        splits = [[122, 30, 99, 84, 5, 97, 148, 52, 136, 71, 96, 121, 89, 115, 33, 80],
                  [17, 0, 93, 151, 102, 57, 108, 3, 50, 13, 51, 79, 36, 66, 144, 137],
                  [105, 85, 81, 43, 131, 152, 74, 73, 76, 132, 138, 127, 25, 147, 39, 48],
                  [10, 111, 55, 27, 63, 150, 29, 72, 77, 141, 58, 4, 130, 40, 106, 41],
                  [117, 46, 65, 126, 6, 100, 101, 153, 16, 95, 157, 62, 91, 155, 78],
                  [44, 135, 37, 22, 88, 60, 70, 11, 8, 149, 23, 129, 156, 24, 83, 35],
                  [125, 59, 1, 110, 128, 45, 38, 133, 92, 154, 68, 12, 9, 109, 21, 64],
                  [47, 53, 82, 124, 67, 14, 20, 19, 103, 75, 69, 26, 140, 94, 61, 142],
                  [87, 120, 54, 18, 146, 90, 31, 32, 123, 104, 28, 116, 112, 139, 143],
                  [113, 107, 98, 134, 119, 145, 2, 56, 49, 42, 15, 34, 86, 7, 114, 118]]


    features = ''
    fold_res = []
    
    overlap = '/home/adam/Documents/lexical_blends_project/project/overlap=True_sample_features'
    noverlap = '/home/adam/Documents/lexical_blends_project/project/overlap=False_sample_features'
    all_blends = '/home/adam/Documents/lexical_blends_project/project/all_sample_features'

    if dataset == 0:
        dpath = overlap
    elif dataset == 1:
        dpath = noverlap
    else:
        dpath = all_blends

    with open(dpath, 'rb') as f:
        features, labels, info = pickle.load(f)
    
    if rem:
        for k, blend in enumerate(features):
            for i, fset in enumerate(blend):
                if isinstance(rem, tuple):
                    for r in rem:
                        del features[k][i][f'f_{r}']
                else:
                    del features[k][i][f'f_{rem}']
    
    for k, test in enumerate(splits):
        if rem:
            p, r, f = run_experiment_on_fold(features, labels, info, overlap, test=test, rem=rem, top_n=top_n, all_s=all_s)
        else:
            p, r, f = run_experiment_on_fold(features, labels, info, overlap, test=test, top_n=top_n, all_s=all_s)
        
        if verbose:
            print('# fold', k, 'results:', p, r, f)
        
        fold_res.append((p, r, f))
    
    all_p, all_r, all_f = 0, 0, 0
    for i, (p, r, f) in enumerate(fold_res):
        all_p += p
        all_r += r
        all_f += f

    return all_p/len(splits), all_r/len(splits), all_f/len(splits)

def ablation_experiment(dataset, test=False, remove=False, top_n=5, overlap=True, strategy=3, all_s=False):

    print('# full fs')
    r1, r2, r3 = cross_val(dataset, overlap=overlap, verbose=False, top_n=top_n, all_s=all_s)
    print(r1, r2, r3)

    # set of features to remove, eg. all 2-grams, not only 2-grams for sw1

    all_fi = set(range(37))
    # categories of features
    if strategy == 1:
        removal_strategy = [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 
                            (12, 13, 14, 15, 16, 17, 20, 23),
                            (18, 19, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32), 
                            (33, 34, 35, 36)]
    elif strategy == 2:
        # groups of features
        removal_strategy = [(0,1,2), (3,4,5), (6,7,8), (9,10,11), (13, 14), 
                            (15, 16), (18, 19, 20), (21, 22, 23), (24, 25), 
                            (26, 27), (28, 29), (30, 31), 32, (33, 35), (34, 36)]
    elif strategy == 3:
        removal_strategy = [(0, 6), (1, 7), (2, 8),
                            (3, 11), (4, 9), (5, 10)]
    elif strategy == 4:
        removal_strategy = [(18,21), (19,22), (20,23)]
    else:
        # # features individually 
        removal_strategy = range(37)

    fdict = feature_indices()
    for k in removal_strategy:
        if isinstance(k, int):
            print('# removing:', fdict[k])
        else:
            print('# removing:', [fdict[y] for y in k])
        #k = tuple(all_fi.difference(set(k)))
        #print(k)
        rn1, rn2, rn3 = cross_val(dataset, overlap=overlap, rem=k, verbose=False, all_s=all_s)
        print(rn1-r1, rn2-r2, rn3-r3)


def run_experiment_on_fold(features, labels, info, overlap=True, test=False, rem=False, top_n=5, all_s=False):
    trainX, trainY = [], []
    devX, devY = [], []
    if not test:
        dev_i = [30, 52, 31, 24, 27, 45, 2, 38, 13]
    else:   
        dev_i = test

    for i, (fs, lb) in enumerate(zip(features, labels)):
        if i in dev_i:
            devX.append(fs)
            devY.append(lb)
        else:
            trainX += fs
            trainY += lb

    # training
    model, FH = train_model(trainX, trainY, resample=False, epochs=1000)
    # testing
    results = test_model(model, FH, devX, devY, top_n=top_n)
    return results

def train_model(dataset, labelset, epochs=1000, resample=False, remove=False):
    #model = LinearRegression()
    model = LogisticRegression()#penalty='l1')
    #model = AdaBoostClassifier(n_estimators=3)

    FH = FeatureHasher()
    dataset = FH.fit_transform(dataset)

    if resample:
        resample_dataset(dataset, labelset)

    return model.fit(dataset, labelset), FH

def test_model_classif():
    pass

def test_model(model, FH, dataset, labelset, top_n=5, all_preds=False):
    label_distribution = defaultdict(int)
    model_results = []

    ### all_classif
    all_pred, all_gold = [], []
    for i, (ds, _labelset) in enumerate(zip(dataset, labelset)):
        blend_results = []
        _dataset = FH.fit_transform(ds)
        
        for dp, lp in zip(_dataset, _labelset):
            label_distribution[lp] += 1
            # select proba for class = True
            _, pred_true = model.predict_proba(dp)[0]
            
            ### all_classif
            pred_label = model.predict(dp)[0]
            all_pred.append(pred_label)

            all_gold.append(lp)
            blend_results.append((pred_true, lp))

        model_results.append(blend_results)

    ### all_classif
    pr = metrics.precision_score(all_gold, all_pred)
    re = metrics.recall_score(all_gold, all_pred)
    fs = metrics.f1_score(all_gold, all_pred)
    ### all_classif
    return pr, re, fs

    #return eval_rank(model_results)
    #return eval_regression(model_results, top_n, rank=True)

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def eval_rank(results):
    mrr = 0
    rs = [] 
    roc_curve = 0
    avgp = 0
    for rset in results:
        if len(rset) == 1:
            rset.append((0, 0))

        ranking = list(reversed(sorted(rset)))
        
        y_pred = [x[0] for x in ranking]
        y_true = [x[1] for x in ranking]
        rs.append(y_true)

        roc_curve += metrics.roc_auc_score(y_true, y_pred)
        avgp += metrics.average_precision_score(y_true, y_pred)

    mrr = mean_reciprocal_rank(rs)
    #print(avgp, roc_curve)
    return avgp/len(results), roc_curve/len(results), mrr          

def eval_regression(results, selection, rank=True, avg=False):
    if rank:
        t, f = 0, 0
        for result_set in results:
            relevant = Counter([x[1] for x in result_set])[1]
            ranking = list(reversed(sorted(result_set)))
            ranking_selection = [x[1] for x in ranking[:selection]]
            if 1 in ranking_selection:
                t += 1
            else:
                f += 1

        return t, f, t/len(results)
    else:
        rdict = {0:0, 1:0, 2:0, 3:0}
        avg_r = []
        for result_set in results:
            ranking = [x[1] for x in list(reversed(sorted(result_set)))]
            gold_r = Counter(ranking)
            rankingd = Counter(ranking[:selection])
            rdict[0] += rankingd[1] # relevant retrieved
            rdict[1] += rankingd[0] # not relevant retrieved
            rdict[2] += gold_r[1]-rankingd[1] # relevant not retrieved
        
        pr = rdict[0]/(rdict[2] + rdict[0])
        re = rdict[0]/(rdict[1] + rdict[0])
        #print(pr, re, 0)
        if pr+re == 0:
            return pr, re, 0
        else:
            return pr, re, 2*((pr*re)/(pr+re))
        
        ## upper bound
        # rdict = {0:0, 1:0, 2:0, 3:0}
        # avg_r = []
        # for result_set in results:
        #     ranking = list(reversed(sorted([x[1] for x in result_set])))
        #     #print(list(reversed(sorted(result_set))))
        #     #ranking = [x[1] for x in list(reversed(sorted(result_set)))]
        #     gold_r = Counter(ranking)
        #     rankingd = Counter(ranking[:selection])
        #     #print(rankingd)
        #     rdict[0] += rankingd[1] # relevant retrieved
        #     rdict[1] += rankingd[0] # not relevant retrieved
        #     rdict[2] += gold_r[1]-rankingd[1] # relevant not retrieved
        
        # pr = rdict[0]/(rdict[2] + rdict[0])
        # re = rdict[0]/(rdict[1] + rdict[0])
        # print(pr, re, 0)
        # return pr, re, 2*((pr*re)/(pr+re))


def eval_log_regression():
    pass
    
if __name__ == '__main__':
    #print(run_experiment_on_fold(all_s=True))
    #print(cross_val(overlap=False, verbose=True, top_n=3, all_s=True))

    #print()

    print(cross_val(2, overlap=True, verbose=True, top_n=3))
    #ablation_experiment(2, overlap=False, top_n=10, strategy=4, all_s=True)



    #for i in [1, 2, 4, 6, 7, 8, 9, 10]:
    #    print('i =', i)
    #    print(cross_val(3, overlap=True, verbose=False, top_n=i, all_s=True))

    # features, labels, info = [], [], []
    # for boolean in [True, False]:
    #     f, l, i = load_all_data(boolean)
    #     with open(f'overlap={boolean}_sample_features', '+wb') as o:
    #         pickle.dump((f, l, i), o)

    #     features += f
    #     labels += l
    #     info += i

    # with open('all_sample_features', '+wb') as o:
    #     pickle.dump((features, labels, info), o)
