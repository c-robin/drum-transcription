## Results ##

**SVM** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 91.4% | 74.1% | 81.9% |
| Snare drum | 93.0% | 79.6% | 85.8% |
| Hi-hat | 84.7% | 93.2% | 88.8% |
| Average | 89.7% | 82.3% | 85.5% |

**K-NN** (k=5)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 82.4% | 82.9% | 82.6% |
| Snare drum | 86.9% | 85.7% | 86.3% |
| Hi-hat | 83.0% | 91.9% | 87.2% |
| Average | 84.1% | 86.9% | 85.4% |

**3 binary SVM** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 90.6% | 76.1% | 82.7% |
| Snare drum | 92.8% | 82.6% | 87.4% |
| Hi-hat | 84.7% | 94.1% | 89.2% |
| Average | 89.4% | 84.3% | 86.4% |

**3 binary SVM, selected features** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 86.6% | 86.2% | 86.4% |
| Snare drum | 90.6% | 86.3% | 88.4% |
| Hi-hat | 82.4% | 94.4% | 88.0% |
| Average | 86.5% | 89.0% | 87.6% |

**3 binary SVM, all features** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 93.4% | 75.9% | 83.7% |
| Snare drum | 95.5% | 82.3% | 88.4% |
| Hi-hat | 84.6% | 97.1% | 90.4% |
| Average | 91.2% | 85.1% | 87.5% |

**3 binary K-NN, all features** (k=5)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Bass drum | 88.5% | 84.9% | 86.6% |
| Snare drum | 91.9% | 90.8% | 91.3% |
| Hi-hat | 91.6% | 92.7% | 92.1% |
| Average | 90.7% | 89.5% | 90.0% |

**Basic-level, SVM** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Average | 87.7% | 65.8% | 72.7% |

**Sub-category, SVM** (C=2, σ=1)
| Instrument | Precision | Recall | F-measure |
|:-----------|:----------|:--------|:----------|
| Average | 81.3% | 44.4% | 52.3% |