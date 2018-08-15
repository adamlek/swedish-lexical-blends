from main import load_all_data
from toolz import keyfilter
from collections import defaultdict, Counter
from statistics import mean, stdev
from math import atan
import random 
import pickle

overlap = True

def collect_f(f, featureset):
    return list(map(lambda y: keyfilter(lambda x: x == f, y), featureset))

def get_mean_sd_from_data():
    features = []
    for overlap_value in [True, False]:
        features += load_all_data(overlap_value)[0]
    #features, *_ = load_all_data(overlap)
    
    baseline = defaultdict(tuple)
    features_ = defaultdict(list)
    
    for l in features:
        for fs in l:
            for k, v in fs.items():
                features_[k].append(v)
    
    for key in features_.keys():
        _mean = mean(features_[key])
        _sd = stdev(features_[key])
        baseline[key] = (_mean, _sd)

    return baseline
        
def rank_samples(baseline, test_model=False):
    features, labels, info = [], [], []
    for overlap_value in [True, False]:
        f, l, i = load_all_data(overlap_value)
        features += f
        labels += l
        info += i
    #features, labels, info = load_all_data(overlap)

    data_scores = []
    for j, blend_list in enumerate(features):
        blend_scores = []
        for i, sample in enumerate(blend_list):
            sample_score = 0
            for k, v in sample.items():
                if baseline[k][1] > 0:
                    if test_model:
                        feature_score = (v-baseline[k][0])/baseline[k][1]
                    else:
                        feature_score = (atan(v)-baseline[k][0])/baseline[k][1]
                else:
                    feature_score = 0

                sample_score += feature_score  
            blend_scores.append((round(sample_score, 3), labels[j][i]))
        data_scores.append(blend_scores)
    
    t, f = 0, 0
    for blend_list in data_scores:
        ranking = [x[1] for x in list(reversed(sorted(blend_list)))]
        n = Counter(ranking)[1]
        n = 1
        ranking = ranking[:n]
        if 1 in ranking:
            t += 1
        else:
            f += 1
    print(t, f, t/(f+t))


def random_baseline_accn(n):

    with open('/home/adam/Documents/lexical_blends_project/project/all_sample_features', 'rb') as f:
        features, labels, info = pickle.load(f)
    # features, labels, info = load_all_data(overlap)

    t, f = 0, len(labels)
    for lset in labels:
        if n>len(lset):
            curr_n = len(lset)
        else:
            curr_n = n

        lset = list(reversed(sorted([(random.random(), int(x)) for x in lset])))
        #print(lset)
        #print(list(map(lambda x: x[1], lset[:curr_n])), len(lset))
        if 1 in list(map(lambda x: x[1], lset[:curr_n])):
            t += 1

    print(t, f, t/f)
            

if __name__ == '__main__':
    #baseline_score = get_mean_sd_from_data()
    #rank_samples(baseline_score, test_model=False)
    for i in [1, 3, 5, 10]:
        random_baseline_accn(i)