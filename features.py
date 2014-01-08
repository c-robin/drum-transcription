import wave, pylab
import yaafelib as yaafe
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

feature_file = 'features.txt'

def mean(array):
    return sum(array)/len(array)
def variance(array, mean_):
    return sum((array-mean_)**2)/len(array)
def normalize(array):
    mean_ = mean(array)
    return (array-mean_)/variance(array, mean_)

class Segment():
    def __init__(self, filename, start_time, end_time):
        self.file = wave.open(filename)
        start = int(44100 * start_time)
        self.file.readframes(start)

        if end_time == -1:
            self.width = -1
        else:
            self.width = int(44100 * (end_time-start_time))

        frames = self.file.readframes(self.width)
        frames = np.fromstring(frames, 'Int16')
        self.frames = np.array(frames, dtype='float')
    def specgram(self):
        pylab.specgram(self.frames, Fs=self.file.getframerate(), scale_by_freq=True)
        plt.show()
    def plot(self):
        x_axis = [float(i) / 44100 for i in range(self.width)]
        plt.plot(x_axis, self.frames)
        plt.show()
    def features(self):
        if len(self.frames) == 0:
            return None

        fp = yaafe.FeaturePlan()
        fp.loadFeaturePlan(feature_file)
        #fp.loadFeaturePlan('features_reduced.txt')

        df = fp.getDataFlow()
        engine = yaafe.Engine()
        engine.load(fp.getDataFlow())
        feats = engine.processAudio(np.array([self.frames]))

        attributes = [mean(values) for (k,values) in sorted(feats.items())]

        return np.concatenate(attributes)

def feature_indices():
    fp = yaafe.FeaturePlan()
    fp.loadFeaturePlan(feature_file)
    df = fp.getDataFlow()
    engine = yaafe.Engine()
    engine.load(fp.getDataFlow())
    indices = sorted([(name, feat['size']) for (name, feat) in engine.getOutputs().items()])
    return sum([[(name, i) for i in range(size)] for (name,size) in indices], [])

def relevant_features(X, Y, d):
    attributes = IRMFSP(X, Y, d=d)
    indices = feature_indices()
    return [indices[i] for i in attributes]
    
# IRMFSP algorithm
# Computes the d most interesting attributes
def IRMFSP(X, Y, d):
    X = X.copy()
    N = len(Y)
    Np = sum(Y)
    Nm = N - Np

    x_s = lambda x,s: np.array([x[s]])
    Mp = lambda s: sum(x_s(X[i], s) for i in range(N) if Y[i] == 1) / float(Np)
    Mm = lambda s: sum(x_s(X[i], s) for i in range(N) if Y[i] == 0) / float(Nm)
    M = lambda s: sum(x_s(X[i], s) for i in range(N)) / float(N)
    
    def r_s(s):
        Ms, Mps, Mms = M(s), Mp(s), Mm(s)
        num = float(Np)/N * norm(Mps - Ms)**2 + float(Nm)/N * norm(Mms - Ms)**2
        denom1 = sum(norm(x_s(X[i], s) - Mps)**2  for i in range(N) if Y[i] == 1)
        denom2 = sum(norm(x_s(X[i], s) - Mms)**2  for i in range(N) if Y[i] == 0)
        denom = denom1/Np + denom2/Nm
        return num / denom

    S = []
    C = range(X.shape[1])

    for j in range(d):
        s_i = C[np.argmax(map(r_s, C))]
        r_i = r_s(s_i)
        S.append(s_i)
        C.remove(s_i)
        for c in C:
            x_c = np.array([xi[c] for xi in X])
            x_si = np.array([xi[s_i] for xi in X])
            new_x_c = x_c - x_si * (np.dot(x_c, x_si)/np.dot(x_si, x_si))
            for i in range(len(X)):
                X[i][c] = new_x_c[i]

    return S

