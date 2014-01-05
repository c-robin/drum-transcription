import wave, pylab
import yaafelib as yaafe
import matplotlib.pyplot as plt
import numpy as np

def mean(array):
    return sum(array)/len(array)
def variance(array, mean_):
    return sum((array-mean_)**2)/len(array)
def normalize(array):
    mean_ = mean(array)
    return (array-mean_)/variance(array, mean_)

# TODO add band-wise energy features, do feature selection
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
        lab.specgram(self.frames, Fs=self.file.getframerate(), scale_by_freq=True)
        plt.show()
    def plot(self):
        x_axis = [float(i) / 44100 for i in range(self.width)]
        plt.plot(x_axis, self.frames)
        plt.show()
    def features(self):
        fp = yaafe.FeaturePlan()
        fp.addFeature('mfcc: MFCC CepsIgnoreFirstCoeff=0 blockSize=2048 stepSize=1024')
        fp.addFeature('shape: SpectralShapeStatistics blockSize=2048 stepSize=1024')
        df = fp.getDataFlow()
        engine = yaafe.Engine()
        engine.load(fp.getDataFlow())
        feats = engine.processAudio(np.array([self.frames]))
        mfcc = feats['mfcc']
        shape = feats['shape']

        if not mfcc.any() or not shape.any():
            return None
       
        mfcc = mean(mfcc)
        shape = mean(shape)

        return np.concatenate((mfcc, shape))
