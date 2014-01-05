#!/usr/bin/python2

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from data import extract_features
from features import mean, variance
from collections import defaultdict
#from random import shuffle
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

# TODO add HMM classifier, configure SVM
if clf == 'knn':
    classifier = KNeighborsClassifier(n_neighbors=3)
else:
    classifier = SVC()

features_, labels_ = extract_features()
features = []
labels = []

# Convert an instrument label into the corresponding category of the taxonomy
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

# Convert the instrument labels, and keep only the values where the labels are covered by the taxonomy
for i, feature in enumerate(features_):
    categories = get_categories(labels_[i])
    if categories is not None:
        features.append(feature)
        labels.append(categories)

# Coverage statistics
print('Statistics\n-----------')
print('Total coverage of this taxonomy: %.1f%%' % (100.0*len(features)/len(features_)))
print('\nCoverage of the different combinations:')
combinations = defaultdict(int)
for l in labels:
    combinations[','.join(l)] += 1
counts = sorted(list(combinations.iteritems()), cmp=lambda x,y: cmp(y[1], x[1]))
total = sum([count for (_,count) in counts])
cover = []
cumul = 0
for l,count in counts:
    cumul += count
    cover.append((l,count,float(cumul)/total))
print('\n'.join(map(str, cover)))

# TODO keep only enough classes for a 95% coverage of all combinations (to avoid under-represented classes)

# Training and evaluation datasets definition
features = np.array(features)
labels = np.array(labels)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)

# TODO training & evaluation from different audio files/drummers

# Normalization of the data
mean_ = mean(X_train)
variance_ = variance(X_train, mean_)
X_train = (X_train - mean_)/variance_
X_test = (X_test - mean_)/variance_

# Training step
classifier.fit(X_train, y_train)

# Evaluation step
# For each instrument:
# [Number of strokes predicted, number of strokes in the groundtruth, number of strokes correctly predicted]
counts = defaultdict(lambda: [0,0,0])

for i, X in enumerate(X_test):
    prediction = classifier.predict(X)[0]
    truth = y_test[i]

    for instr in prediction:
        counts[instr][0] += 1

        if instr in truth:
            counts[instr][2] += 1

    for instr in truth:
        counts[instr][1] += 1

print('\nEvaluation\n----------')
scores = dict()
for instr, (n_predict, n_truth, n_correct) in counts.iteritems():
    precision = float(n_correct)/n_predict
    recall = float(n_correct)/n_truth
    
    if precision == 0 or recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision + recall)
    
    scores[instr] = (precision, recall, f_measure)

    print('%s: precision=%.3f, recall=%.3f, F-measure=%.3f' % (instr, precision, recall, f_measure))

"""
if cross_val:
    score = cross_validation.cross_val_score(classifier, features, labels, cv=10)
    score = mean(score)
else:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
#print(score)
"""

