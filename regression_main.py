from helper_functions import get_blends_csv, lev
import numpy as np
from itertools import chain
from collections import defaultdict, Counter
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import LinearSVC, SVC
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

def featureset_modification(featureset, remove):
    if isinstance(remove, list):
        for f in remove:
            del featureset[f]
    else:
        del featureset[remove]
    return featureset

def lexicon_features(fset, blend, cw1, cw2, lexicon_dict):

    sw1_d = keyfilter(lambda x: x == cw1, lexicon_dict)
    mother_entry = []
    father_entry = []
    pos_entry = []
    for k, v in sw1_d.items():
        mother_entry.append(v[1])
        father_entry.append(v[2])
        pos_entry.append(v[0])
    
    #print(pos_entry)
        
    fset['sw1_pos'] = '_'.join(pos_entry)
    #fset['sw1_father'] = '_'.join(father_entry)
    #fset['sw1_mother'] = '_'.join(mother_entry)

    sw2_d = keyfilter(lambda x: x == cw2, lexicon_dict)
    mother_entry = []
    father_entry = []
    pos_entry = []
    for k, v in sw2_d.items():
        mother_entry.append(v[1])
        father_entry.append(v[2])
        pos_entry.append(v[0])

    fset['sw2_pos'] = '_'.join(pos_entry)
    #fset['sw2_father'] = '_'.join(father_entry)
    #fset['sw2_mother'] = '_'.join(mother_entry)

    return fset
    
def add_lexicon_data():
    folder = '/home/adam/Documents/lexical_blends_project/data/'
    with open(folder+f'saldo_lex.pickle', 'rb') as f:
        lexicon_dict = pickle.load(f)
    fs, lb, info = load_all_data()

    for i, blend in enumerate(fs):
        for j, vec in enumerate(blend):
            fs[i][j] = lexicon_features(vec, info[i][0], info[i][1], info[i][2], lexicon_dict)
            
    with open('with_saldo_data.pickle', '+wb') as f:
        pickle.dump((fs, lb, info), f)

def resample_dataset(dataset, labelset):
    d = Counter(labelset)
    over_sampling = SMOTE(k_neighbors=2, n_jobs=4, ratio={0:d[0], 1:d[1]*10})
    #under_sampling = AllKNN(n_jobs=4, n_neighbors=2)
    #combined_sampling = SMOTETomek()
    dataset, labelset = over_sampling.fit_sample(dataset, labelset)
    print('## train resample distribution', Counter(labelset))
    return dataset, labelset

def load_all_data(remove=False, pos=False, sampa=False):
    #fpath = '/home/adam/Documents/lexical_blends_project/project/saldo_features_overlap_blends.csv'
    fpath = '/home/adam/Documents/lexical_blends_project/project/saldo_features_overlap_split_blends_260718.csv'

    lexicon = 'saldo'

    folder = '/home/adam/Documents/lexical_blends_project/data/'
    with open(folder+f'{lexicon}_lex.pickle', 'rb') as f:
        lexicon_dict = pickle.load(f)

    with open('/home/adam/Documents/lexical_blends_project/data/nst_lex.pickle', 'rb') as f:
        nst = pickle.load(f)

    T, F = 0,0
    current_blend = 'göteburgare'
    #print('# extracting saldo data for', current_blend)
    with open(fpath) as f:
        featureset = [[]]
        labelset = [[]]
        infoset = [[]]
        for ln in f:
            *features, label, bl, w1, w2, w1_split, w2_split = ln.split(',')
            #entry = ln.split(',')

            fset = dict([(f'f_{i}', float(x)) for i, x in enumerate(features)])
            
            if sampa:
                if w1 in nst and w2 in nst:
                    w1_sampa = nst[w1][1]
                    w2_sampa = nst[w2][1]
                    fset['w1w2_lev'] = lev(w1_sampa, w2_sampa)
            if pos:
                fset['w1_pos'] = lexicon_dict[w1][0]
                fset['w2_pos'] = lexicon_dict[w2][0]

            if label == 'True':
                label = 1
                T += 1
            else:
                label = 0
                F += 1
            
            if remove:
                fset = featureset_modification(fset, remove)
            
            #fset = lexicon_features(fset, bl, w1, w2, lexicon_dict)

            if bl == current_blend:
                featureset[-1].append(fset)
                infoset[-1].append((bl, w1, w2))
                #featureset[-1].append(np.array([float(x) for x in features]))
                labelset[-1].append(label)
            else:
                current_blend = bl
                featureset.append([fset])
                infoset.append([(bl, w1, w2)])
                #featureset.append([np.array([float(x) for x in features])])
                labelset.append([label])

    return featureset, labelset, infoset

def cross_val_linear_regression(rem=False):
    splits = [[30, 52, 31, 24, 27, 45, 2, 38, 13],
              [4, 33, 19, 40, 56, 62, 60, 57, 21],
              [47, 48, 53, 10, 6, 14, 42, 39, 35],
              [5, 20, 28, 9, 22, 58, 16, 36, 15],
              [51, 1, 34, 55, 7, 44, 32, 49, 25],
              [50, 17, 12, 46, 61, 11, 3, 23, 54],
              [59, 26, 37, 18, 41, 43, 29, 0, 8]]

    fold_res = []
    
    for k, test in enumerate(splits):
        if rem:
            p, r, f = dev(test, linear=True, rem=rem)
        else:
            p, r, f = dev(test, linear=True)
        fold_res.append((p, r, f))
    
    all_p, all_r, all_f = 0, 0, 0
    for i, (p, r, f) in enumerate(fold_res):
        if i == 0:
            continue
        all_p += p
        all_r += r
        all_f += f

    #print('avg_p', all_p/6)
    #print('avg_r', all_r/6)
    #print('avg_f', all_f/6)
    return all_p/6, all_r/6, all_f/6

def cross_val():
    splits = [[30, 52, 31, 24, 27, 45, 2, 38, 13],
              [4, 33, 19, 40, 56, 62, 60, 57, 21],
              [47, 48, 53, 10, 6, 14, 42, 39, 35],
              [5, 20, 28, 9, 22, 58, 16, 36, 15],
              [51, 1, 34, 55, 7, 44, 32, 49, 25],
              [50, 17, 12, 46, 61, 11, 3, 23, 54],
              [59, 26, 37, 18, 41, 43, 29, 0, 8]]

    fold_res = []
    
    for k, test in enumerate(splits):
        p, r, f = dev(test, linear=True)
        fold_res.append((p, r, f))
    
    all_p, all_r, all_f = 0, 0, 0
    for i, (p, r, f) in enumerate(fold_res):
        if i == 1:
            continue
        all_p += p
        all_r += r
        all_f += f

    print('avg_p', all_p/6)
    print('avg_r', all_r/6)
    print('avg_f', all_f/6)


def dev_ablation(test=False, remove=False):

    print('# full fs')
    r1, r2, r3 = cross_val_linear_regression()
    print(r1, r2, r3)

    # set of features to remove, eg. all 2-grams, not only 2-grams for sw1
    #
    removal_strategy = [(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
                        (21,22,23,24,25,26,27,28,29,30)]

    for k in range(38):
        print('# removing', k)
        r = f'f_{k}'
        rn1, rn2, rn3 = cross_val_linear_regression(rem=r)
        print(rn1-r1, rn2-r2, rn3-r3)


def dev(test=False, linear=False, rem=False):
    trainX, trainY = [], []
    devX, devY = [], []
    if not test:
        dev_i = [30, 52, 31, 24, 27, 45, 2, 38, 13]
    else:   
        dev_i = test
    
    #with open('/home/adam/Documents/lexical_blends_project/project/with_saldo_data.pickle', 'rb') as f:
    #    features, labels, info = pickle.load(f)
    if rem:
        features, labels, info = load_all_data(remove=rem)
    else:
        features, labels, info = load_all_data()

    for i, (fs, lb) in enumerate(zip(features, labels)):
        if i in dev_i:
            devX.append(fs)
            devY.append(lb)
        else:
            trainX += fs
            trainY += lb

    if linear:
        #print('### training:')
        model, FH = train_linear_regression_model(trainX, trainY, resample=False, epochs=1000)

        #print('### testing:')
        results = test_linear_regression_model(model, FH, devX, devY)
        #print('# results = ', results)
    else:
        print('### training:')
        model, FH = train_regression_model(trainX, trainY, resample=False, epochs=1000)

        print('### testing:')
        results = test_regression_model(model, FH, devX, devY)
        print('# results = ', results) 
    #correct, incorrect = eval_regression(results)

    return results

def transform_vectors(list_of_dicts):
    dataset = []
    for d in list_of_dicts:
        v = np.array([float(x) for x in d.values()]).reshape(1,-1)
        dataset.append(v)
    return np.array(dataset)

def train_regression_model(dataset, labelset, epochs=1000, resample=False, remove=False):
    label_distribution = Counter(labelset)

    classif = Perceptron()
    #classif = LogisticRegression(solver='liblinear', penalty='l1')#, class_weight='balanced')

    #print('## train sample distribution', label_distribution)

    FH = FeatureHasher()
    dataset = FH.fit_transform(dataset)
    #dataset = transform_vectors(dataset)
    #samp, nx, ny = dataset.shape
    #dataset = dataset.reshape((samp, nx*ny))

    if resample:
        resample_dataset(dataset, labelset)

    return classif.fit(dataset, labelset), FH

def test_regression_model(model, FH, dataset, labelset):
    label_distribution = defaultdict(int)
    model_results = []
    trueclass_acc = [0 for x in labelset]
    y_gold, y_pred = [], []

    for i, (ds, _labelset) in enumerate(zip(dataset, labelset)):
        blend_results = []
        _dataset = FH.fit_transform(ds)
        #_dataset = transform_vectors(ds)
        
        for dp, lp in zip(_dataset, _labelset):
            label_distribution[lp] += 1
            pred = model.predict(dp)[0]
            blend_results.append((pred, lp))
            y_gold.append(lp)
            y_pred.append(pred)
        model_results.append(blend_results)

    #print('# test sample distribution', label_distribution)

    prec = metrics.precision_score(y_gold, y_pred)
    rec = metrics.recall_score(y_gold, y_pred)
    fscore = metrics.f1_score(y_gold, y_pred)

    return prec, rec, fscore 
    #return model_results

def train_linear_regression_model(dataset, labelset, epochs=1000, resample=False, remove=False):
    label_distribution = Counter(labelset)

    classif = LinearRegression()

    FH = FeatureHasher()
    dataset = FH.fit_transform(dataset)

    if resample:
        resample_dataset(dataset, labelset)

    return classif.fit(dataset, labelset), FH

def test_linear_regression_model(model, FH, dataset, labelset):
    label_distribution = defaultdict(int)
    model_results = []

    #print(model.coef_)

    for i, (ds, _labelset) in enumerate(zip(dataset, labelset)):
        blend_results = []
        _dataset = FH.fit_transform(ds)
        #_dataset = transform_vectors(ds) 
        
        for dp, lp in zip(_dataset, _labelset):
            #print(dp.shape)
            label_distribution[lp] += 1
            pred = model.predict(dp)[0]
            blend_results.append((pred, lp))

        model_results.append(blend_results)

    return eval_regression(model_results, 3, False)

def eval_regression(results, selection, rank=False, avg=False):
    if rank:
        t, f = 0, 0
        for result_set in results:
            relevant = Counter([x[1] for x in result_set])[1]
            ranking = list(reversed(sorted(result_set)))
            ranking_selection = [x[1] for x in ranking[:selection]]
            #print(ranking_selection, relevant)
            #print(ranking_selection)
            if 1 in ranking_selection:
                t += 1
            else:
                f += 1
            #print(ranking[:10])
        return t, f, t/len(results)
    else:
        rdict = {0:0, 1:0, 2:0, 3:0}
        avg_r = []
        for result_set in results:
            #print(list(reversed(sorted(result_set))))
            ranking = [x[1] for x in list(reversed(sorted(result_set)))]
            gold_r = Counter(ranking)
            rankingd = Counter(ranking[:selection])
            #print(rankingd)
            rdict[0] += rankingd[1] # relevant retrieved
            rdict[1] += rankingd[0] # not relevant retrieved
            rdict[2] += gold_r[1]-rankingd[1] # relevant not retrieved
        
        pr = rdict[0]/(rdict[2] + rdict[0])
        re = rdict[0]/(rdict[1] + rdict[0])
        #print(pr, re, 0)
        return pr, re, 2*((pr*re)/(pr+re))
        
        ###
        # UPPER BOUND
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
    
    if avg:
        for result_set in results:
            #print(list(reversed(sorted(result_set))))
            ranking = [x[1] for x in list(reversed(sorted(result_set)))]
            number_of_items = len(ranking)

def eval_log_regression():
    pass
            

            

if __name__ == '__main__':
    #print(dev(linear=True))
    #print(cross_val_linear_regression())
    #dev_ablation()
    #cross_val()

    a, b, c = load_all_data()
    print(a[0][0])