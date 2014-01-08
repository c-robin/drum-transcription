from os import path, listdir, getuid
from features import Segment
from random import shuffle
from pwd import getpwuid

if 'zycho' in getpwuid(getuid())[4]:
    root = '/home/zycho/workspace/FAV/ENST-Drums-Audio'
else:
    root = '/media/windows/Users/eske/ENST-Drums-Audio'
drummers = ['drummer_1', 'drummer_2', 'drummer_3']
audio_subdir = 'audio/overhead_L'
txt_subdir = 'annotation'

def extract_features():
    features = []
    labels = []

    files = get_files('minus-one')
    shuffle(files)
    #files = files[:10]

    for txt_file, audio_file in files:
        for (start_time, end_time, labels_) in extract_segments(txt_file):
            segment = Segment(audio_file, start_time, end_time)
            feature = segment.features()

            if feature is None:
                continue

            features.append(feature)
            labels.append(labels_)

    return features, labels

def get_files(name_pattern):
    files = []
    for drummer in drummers:
        audio_dir = path.join(root, drummer, audio_subdir)
        txt_dir = path.join(root, drummer, txt_subdir)

        for file_ in listdir(audio_dir):
            name = file_[:-4]
            if name_pattern not in name:
                continue
            txt_file = path.join(txt_dir, name + '.txt')
            audio_file = path.join(audio_dir, file_)
            files.append((txt_file, audio_file))
    return files

def extract_segments(txt_file):
    events = []
    with open(txt_file) as file_:
        lines = file_.readlines()

        for line in lines:
            time, label = line[:-1].split(' ')
            events.append((float(time), label))

    segments = []
    cur_labels = []
    cur_time = None
    last_time = None

    for time,label in events:
        if cur_time is None:
            cur_labels.append(label)
            cur_time = time
        # strokes appearing within a window of 50ms are likely to be simultaneous strokes
        elif time < cur_time + 0.05:
            cur_labels.append(label)
        else:
            # otherwise the segment is not long enough for proper analysis
            if time - cur_time >= 0.1:
                segments.append((cur_time, min(time, cur_time + 0.2), cur_labels))
            cur_time = time
            cur_labels = [label]

    return segments
