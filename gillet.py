#!/usr/bin/python2

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn import cross_validation
from data import extract_features
from features import mean, variance, normalize
from collections import defaultdict
import numpy as np

# Simple taxonomy, with 75% coverage
taxonomy = {
    'bass_drum': ['bd'],
    'snare_drum': ['sd', 'sd-'],
    'hihat': ['ohh', 'chh'],
}

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

x_values = defaultdict(list)
y_values = defaultdict(list)

for i, feature in enumerate(features):
    instruments = labels[i]

    for instr in taxonomy.keys():
        x_values[instr].append(feature)
        y_values[instr].append(1 if instr in instruments else 0)

def score_func(y_true, y_pred):
    return (precision_score(y_true,y_pred),
            recall_score(y_true,y_pred),
            f1_score(y_true,y_pred))

for instr in x_values:
    X = np.array(x_values[instr])
    Y = np.array(y_values[instr])
    
    C = 2
    d = X.shape[1]
    delta = 1
    gamma = 1.0/(2*d*delta**2)
    clf = SVC(C=C, gamma=gamma)

    scores = cross_validation.cross_val_score(clf,X,Y,score_func,cv=10)
    precision,recall,f1 = mean(scores)
    support = sum(Y)
    print('%s: precision=%.3f, recall=%.3f, F-measure=%.3f, support=%d' % (instr, precision, recall, f1, support))
