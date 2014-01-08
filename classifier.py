#!/usr/bin/python2
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.hmm import GaussianHMM
from data import extract_features
from features import normalize, mean
from collections import defaultdict
from numpy.linalg import norm
import numpy as np
import sys
from pprint import pprint
import time


# Tree levels of taxonomies, with 95% coverage
super_category = {
    'membrane': ['lft', 'sd-', 'lt', 'bd', 'lmt', 'mt', 'sd'],
    'plate': ['ohh', 'cb', 'ch1', 'cr2', 'cr1', 'cr5', 'chh', 'spl2', 'c1', 'c4', 'rc4', 'rc2', 'rc3', 'ch5'],
}

basic_level = {
    'Bass drum': ['bd'],
    'Snare drum': ['sd', 'sd-'],
    'Tom': ['lft', 'lt', 'lmt', 'mt'],
    'Hi-hat': ['ohh', 'chh'],
    'Cymbal': ['ch1', 'cr2', 'cr1', 'cr5', 'spl2', 'c1', 'c4', 'rc4', 'rc2', 'rc3', 'ch5'],
}

sub_category = {
    'Bass drum': ['bd'],
    'Snare drum': ['sd', 'sd-'],
    'Lowest tom': ['lft'],
    'Low tom': ['lt'],
    'Low mid tom': ['lmt'],
    'Mid tom': ['mt'],
    'Open hi-hat': ['ohh'],
    'Closed hi-hat': ['chh'],
    'Ride cymbal': ['ch1', 'rc4', 'rc2', 'rc3', 'ch5'],
    'Crash cymbal': ['cr2', 'cr1', 'cr5'],
    'Splash cymbal': ['spl2'],
    'Other cymbal': ['c1', 'c4'],
}

# Simple taxonomy, with 75% coverage
gillet = {
    'Bass drum': ['bd'],
    'Snare drum': ['sd', 'sd-'],
    'Hi-hat': ['ohh', 'chh'],
}

taxonomies = {'gillet': gillet, 'super': super_category, 'basic': basic_level, 'sub': sub_category}

if len(sys.argv) != 3:
    sys.exit('Usage: %s taxonomy classifier' % sys.argv[0])
_, tax, clf = sys.argv

try:
    taxonomy = taxonomies[tax]
except:
    sys.exit('category must be one of these: %s' % str(list(taxonomies.keys())))

classifiers = ['knn', 'svm', '3svm','hmm']
if clf not in classifiers:
    sys.exit('classifier must be one of these: %s' % str(classifiers))

features_, labels_ = extract_features()
features = []
labels = []

# Convert an instrument label into the corresponding taxonomy category
def get_category(label):
    for category_, labels in taxonomy.iteritems():
        if label in labels:
            return category_
    return None
# Convert a list of instrument labels into a list of taxonomy categories
def get_categories(labels):
    categories = set([])
    for label in labels:
        category_ = get_category(label)
        if category_ is None:
            return None
        else:
            categories.add(category_)
    return sorted(list(categories))

# Convert the instrument labels, and keep only the segments whose instruments are all covered by the taxonomy
for i, feature in enumerate(features_):
    categories = get_categories(labels_[i])
    if categories is not None:
        features.append(feature)
        labels.append(categories)

features = np.array(features)
features = normalize(features)
labels = np.array(labels)

# Result summary
sum_scores = dict()

# (C, sigma)
svm_params = (2, 1)
#params = {
#    'Hi-hat': (4, 1),
#    'Snare drum': (2, 2),
#    'Bass drum': (2, 4),
#}

if clf == '3svm':
    x_values = defaultdict(list)
    y_values = defaultdict(list)
    for i, feature in enumerate(features):
        instruments = labels[i]
        for instr in taxonomy.keys():
            x_values[instr].append(feature)
            y_values[instr].append(int(instr in instruments))

    def score_func(y_true, y_pred):
        return (precision_score(y_true,y_pred),
                recall_score(y_true,y_pred),
                f1_score(y_true,y_pred))
        
    for instr in x_values:
        X = np.array(x_values[instr])
        Y = np.array(y_values[instr])
        
        #MERDE DE CHARLES+ESKE
        N = len(Y)
        Np = sum(Y)
        Nm = len(Y)-Np

        x_s = lambda x,s: np.array([xi for (i,xi) in enumerate(x) if i in s])
        Mp = lambda s: sum(x_s(X[i],s) for i in range(N) if Y[i]==1)/Np
        Mm = lambda s: sum(x_s(X[i],s) for i in range(N) if Y[i]==0)/Nm
        M = lambda s: sum(x_s(X[i],s) for i in range(N))/N
        r_s = lambda s: (Np/N*norm(Mp(s)-M(s))**2 + Nm/N*norm(Mm(s)-M(s))**2)/((sum(norm(x_s(X[i],s)-Mp(s))**2  for i in range(N) if Y[i]==1)/Np)+(sum(norm(x_s(X[i],s)-Mm(s))**2  for i in range(N) if Y[i]==0)/Nm))
        #ALGO IRMFSP
        #calcul des d features les plus interessante
        d=4
        def IRMFSP(X,Y,d):
            S = set([])
            C = range(len(X[0]))
            j = 0
            base = time.clock()
            while j<d:
                print 'j'+str(j)+' '+str(time.clock()-base)
                s_i = np.argmax(map(r_s, [set([c]) for c in C]))
                r_i = r_s(set([s_i]))
                S.add(s_i)
                if s_i in C: C.remove(s_i)
                else: sys.exit('IRMFSP : erreur à C prive de s_i')
                for c in C:
                    print 'c'+str(c)+' '+str(time.clock()-base)
                    x_c = np.array([xi[c] for xi in X])
                    x_si = np.array([xi[s_i] for xi in X])
                    new_x_c = x_c - np.dot(x_c, x_si)/np.dot(x_si, x_si)*x_si
                    for i in range(len(X)):
                        print 'i'+str(i)+' '+str(time.clock()-base)
                        X[i][c] = new_x_c[i] 
                j=j+1
            return S
        pprint(IRMFSP(X,Y,d))

        #FIN MERDE DE CHARLES+ESKE
        d = X.shape[1]
        C, sigma = svm_params
        gamma = 1.0/(2*d*sigma**2)
        classifier = SVC(C=C, gamma=gamma)

        scores = cross_validation.cross_val_score(classifier, X, Y, score_func, cv=10)
        sum_scores[instr] = 100*mean(scores)
elif clf == 'hmm':
    pprint.pprint(features)
    pprint.pprint(labels)
    X = np.column_stack([features,labels])
    classifier = GaussianHMM(len(taxonomy), covariance_type="diag", n_iter=1000)
    classifier.fit(features)

    def score_func(y_true, y_pred):
        return  classifier.predict(y_true)

    scores = cross_validation.cross_val_score(classifier, features, labels, score_func, cv=10)
else:
    if clf == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif clf == 'svm':
        d = features.shape[1]
        C, sigma = svm_params
        gamma = 1.0/(2*d*sigma**2)
        classifier = SVC(C=C, gamma=gamma)

    def score_func(y_true, y_pred):
        counts = defaultdict(lambda: [0,0,0])
        for i, truth in enumerate(y_true):
            prediction = y_pred[i]
            for instr in prediction:
                counts[instr][0] += 1
                if instr in truth:
                    counts[instr][2] += 1
            for instr in truth:
                counts[instr][1] += 1
        scores = dict()
        for instr, (n_predict, n_truth, n_correct) in counts.iteritems():
            if n_predict == 0:
                precision = 1
            else:
                precision = float(n_correct)/n_predict
            recall = float(n_correct)/n_truth
            if precision == 0 or recall == 0:
                f_measure = 0
            else:
                f_measure = 2*precision*recall/(precision + recall)
            scores[instr] = np.array([precision, recall, f_measure])
        return scores

    scores = cross_validation.cross_val_score(classifier, features, labels, score_func, cv=10)

    # Compute the means of the scores over all folds
    for scores_ in scores:
        for instr, values in scores_.iteritems():
            values *= 100.0/len(scores)
            if instr not in sum_scores:
                sum_scores[instr] = values
            else:
                sum_scores[instr] += values

# Display the results in human readable form, google wiki markup, and latex
def latex_row(info):
    return '%s&%.1f\\%%&%.1f\\%%&%.1f\\%%\\\\\n\\hline\n' % info 
def wiki_row(info):
   return '|| %s || %.1f%% || %.1f%% || %.1f%% ||\n' % info

latex_table = '\\begin{tabular}{|c|c|c|c|}\n\\hline\nInstrument&Precision&Recall&F-measure\\\\\n\\hline\n'
wiki_table = '|| Instrument || Precision || Recall || F-measure ||\n'

average = np.array([0.0, 0.0, 0.0])
for instr, values in sum_scores.iteritems():
    average += values/len(sum_scores)
    info = (instr,) + tuple(values)

    print('%s: precision=%.1f%%, recall=%.1f%%, F-measure=%.1f%%' % info)

    latex_table += latex_row(info)
    wiki_table += wiki_row(info)

latex_table += latex_row(('Average',) + tuple(average))
wiki_table += wiki_row(('Average',) + tuple(average))
latex_table += '\\end{tabular}'
print('\n' + wiki_table)
print(latex_table)
