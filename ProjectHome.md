## This work ##
This work is about the automatic transcription of drum sequences. We use the audio tracks from the [ENST-Drums database](http://www.tsi.telecom-paristech.fr/aao/en/2010/02/19/enst-drums-an-extensive-audio-visual-database-for-drum-signals-processing/).

We distinguish 4 different taxonomies of instruments:
  * _Gillet's taxonomy_: Bass drum, snare drum, hi-hat;
  * _Super-category_: Membrane, plate;
  * _Basic-level_: Bass drum, snare drum, tom, hi-hat, cymbal;
  * _Sub-category_: Bass drum, snare drum, lowest tom, low tom, low mid tom, mid tom, open hi-hat, closed hi-hat, ride cymbal, crash cymbal, splash cymbal, other cymbal.

_Super-category_, _basic-level_, and _sub-category_ were defined by Herrera et al. in (4).
_Gillet's taxonomy_ is the taxonomy used by Gillet in his thesis (1).

For most of this work, we placed ourselves in similar conditions as in Gillet's thesis.

We use the _midi-one_ tracks of the database, without accompaniment (i.e. noise -∞ db). We focus mainly on the _Gillet_ taxonomy, in order to be able to compare our results with his results.

### Segmentation ###
In his thesis, Gillet does _onset detection_ in order to detect when a stroke occurs. In our case, onset detection isn't really the point of the exercise. That is why we use the ground truth as an oracle to extract the segments corresponding to strokes.

The difficulty is when several instruments are played (nearly) simultaneously. We consider that instruments which are played within the same window of 50 ms belong to the same segment.

### Features ###
We use the same features as in (2):
  * The means of the first 13 MFCC coefficients, using an analysis window of 2048 frames, with a 50% overlap;
  * 4 spectral shape parameters as defined by Gillet: spectral centroïd, width, skewness, and kurtosis. This feature is called _SpectralShapeStatistics_ in Yaafe;
  * The last feature, _band-wise frequency content parameters_ doesn't exist in Yaafe, so we don't use it for now.

The set _all features_ contains a more exhaustive list of features. The set _selected features_, contains features collected by the _IRMFSP_ algorithm.

### Classification ###
There are two ways of handling classification:
  * One single classifier, with 2<sup>N</sup> classes (one for each combination of instruments).
  * Or N binary classifiers, with N classes (one for each instrument), that decides if the instrument is played or not.

We try two different classifiers: a SVM classifier with RBF kernel as in (2), and a K-NN as in (4).

### Results ###
**Gillet, SVM** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 91.4% | 74.1% | 81.9% |
| Snare drum | 93.0% | 79.6% | 85.8% |
| Hi-hat | 84.7% | 93.2% | 88.8% |
| Average | 89.7% | 82.3% | 85.5% |

**Basic-level, SVM** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Average | 87.7% | 65.8% | 72.7% |

**Sub-category, SVM** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Average | 81.3% | 44.4% | 52.3% |

**Gillet, 3 binary K-NN, all features** (k=5)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 86.2% | 87.5% | 86.8% |
| Snare drum | 89.8% | 92.7% | 91.2% |
| Hi-hat | 89.7% | 94.8% | 92.2% |
| Average | 88.5% | 91.7% | 90.1% |

For more results, visit [this page.](Results.md)

## Bibliography ##
  1. O. Gillet. _Transcription des signaux percussifs. Application à l'analyse de scènes musicales audiovisuelles._ PhD thesis, 2007.
  1. O. Gillet and G. Richard. _Automatic transcription of drum loops._ 2004.
  1. O. Gillet and G. Richard. _Enst-drums: an extensive audio-visual database for drum signals processing._ 2006.
  1. P. Herrera, A. Yeterian, R. Yeterian and F. Gouyon. _Automatic Classification of Drum Sounds: A Comparison of Feature Selection and Classification Techniques._ 2002.

Gillet's thesis is quite big, so here is a small index (the section that is interesting for us is the section 4).
  * Taxonomy description: p.54
  * Attribute extraction: p.61
  * Attribute selection: p.65 and p.91
  * Classification parameters: p.67 and p.93
  * HMM: p.74
  * Database: p.84
  * Classification results: p.88

## Dependencies ##
  * Yaafe, http://yaafe.sourceforge.net/
  * Scikit-learn, http://scikit-learn.org/

## Python resources ##
  * Machine learning, http://scikit-learn.org/
  * Features extraction, http://yaafe.sourceforge.net/
  * Wave audio files, http://docs.python.org/2/library/wave.html
  * Signal processing, http://docs.scipy.org/doc/scipy/reference/signal.html
  * Plot drawing, http://matplotlib.org/
  * Feature selection, http://scikit-learn.org/stable/modules/feature_selection.html
  * Onset detection tool, https://github.com/johnglover/modal