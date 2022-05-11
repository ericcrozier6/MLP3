import librosa as lb
import os
import matplotlib.pyplot as plt

#script reads in the first 29s of each mp3 in the test set and converts them to spectrograms and saves them to their own folder.
for fileName in os.scandir(os.path.abspath("./MusicData/test")):
    data, FS = lb.load(os.path.abspath(fileName), duration=29)
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs = FS, noverlap=0, NFFT=128)
    ax.axis('off')
    specFileName = os.path.abspath(os.path.abspath("./testSpecGrams") + '/' + fileName.name[:len(fileName.name)-3] + "png")
    fig.savefig(specFileName, dpi=300)