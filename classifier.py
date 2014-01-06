#!/usr/bin/python2

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from data import extract_features
from features import normalize
from collections import defaultdict
import numpy as np
import sys

# Tree levels of taxonomies, with 95% coverage
super_category = {
    'membrane': ['lft', 'sd-', 'lt', 'bd', 'lmt', 'mt', 'sd'],
    'plate': ['ohh', 'cb', 'ch1', 'cr2', 'cr1', 'cr5', 'chh', 'spl2', 'c1', 'c4', 'rc4', 'rc2', 'rc3', 'ch5'],
}

basic_level = {
    'bass_drum': ['bd'],
    'snare_drum': ['sd', 'sd-'],
    'tom': ['lft', 'lt', 'lmt', 'mt'],
    'hihat': ['ohh', 'chh'],
    'cymbal': ['ch1', 'cr2', 'cr1', 'cr5', 'spl2', 'c1', 'c4', 'rc4', 'rc2', 'rc3', 'ch5'],
}

sub_category = {
    'bass_drum': ['bd'],
    'snare_drum': ['sd', 'sd-'],
    'lowest_tom': ['lft'],
    'low_tom': ['lt'],
    'low_mid_tom': ['lmt'],
    'mid_tom': ['mt'],
    'open_hihat': ['ohh'],
    'closed_hihat': ['chh'],
    'ride_cymbal': ['ch1', 'rc4', 'rc2', 'rc3', 'ch5'],
    'crash_cymbal': ['cr2', 'cr1', 'cr5'],
    'splash_cymbal': ['spl2'],
    'other_cymbal': ['c1', 'c4'],
}

# Simple taxonomy, with 75% coverage
gillet = {
    'bass_drum': ['bd'],
    'snare_drum': ['sd', 'sd-'],
    'hihat': ['ohh', 'chh'],
}

taxonomies = {'gillet': gillet, 'super': super_category, 'basic': basic_level, 'sub': sub_category}

if len(sys.argv) != 3:
    sys.exit('Usage: %s taxonomy classifier' % sys.argv[0])
_, tax, clf = sys.argv

try:
    taxonomy = taxonomies[tax]
except:
    sys.exit('category must be one of these: %s' % str(list(taxonomies.keys())))

classifiers = ['knn', 'svm']
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

# TODO add HMM classifier, configure SVM
if clf == 'knn':
    classifier = KNeighborsClassifier(n_neighbors=3)
elif clf == 'svm':
    C = 2
    d = features.shape[1]
    sigma = 1
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
        precision = float(n_correct)/n_predict
        recall = float(n_correct)/n_truth
        if precision == 0 or recall == 0:
            f_measure = 0
        else:
            f_measure = 2*precision*recall/(precision + recall)
        scores[instr] = np.array([precision, recall, f_measure])
    return scores

scores = cross_validation.cross_val_score(classifier,features,labels,score_func,cv=10)

# Compute the means of the scores over all test folds
sum_scores = dict()
for scores_ in scores:
    for instr, values in scores_.iteritems():
        values /= len(scores)
        if instr not in sum_scores:
            sum_scores[instr] = values
        else:
            sum_scores[instr] += values

for instr, (precision, recall, f_measure) in sum_scores.iteritems():
    print('%s: precision=%.3f, recall=%.3f, F-measure=%.3f' % (instr, precision, recall, f_measure))

