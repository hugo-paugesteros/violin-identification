import scipy.signal
import librosa
import matplotlib.pyplot as plt
import numpy as np

file = f'../../Data/SegmentationPerViolin/PLAYER5/Violin5.wav'
y, sr = librosa.load(file)

(f, Pxx) = scipy.signal.welch(y, fs=sr, nperseg=4096, noverlap=2048, average='mean')
(f2, Pxx2) = scipy.signal.periodogram(y, fs=sr)
(f3, t, Zxx) = scipy.signal.stft(x=y, fs=sr, nperseg=4096, noverlap=2048, scaling='psd')
Pxx3 = np.mean(np.power(np.abs(Zxx), 2), axis=1)

plt.plot(f, Pxx, alpha=.5)
plt.plot(f2, Pxx2, alpha=.5)
plt.plot(f3, Pxx3, alpha=.5)
plt.show()