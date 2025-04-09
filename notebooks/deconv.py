import pandas as pd
import librosa
import numpy as np
from features import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

h, sr = librosa.load('/home/hugo/Thèse/Data/Villefavard 2024/room.mp3')
y, sr = librosa.load('/home/hugo/Thèse/Data/Villefavard 2024/Violin 5/ellin.mp3', offset=5, duration=5)

scipy.io.wavfile.write('orig.wav', sr, y)
N = int(2**(np.ceil(np.log2(len(h) + len(y)))))
print(N)
filtered = np.real(np.fft.ifft(np.fft.fft(y, N) / (np.fft.fft(h, N) + 5e2)))
filtered /= np.max(np.abs(filtered))
scipy.io.wavfile.write('test.wav', sr, filtered)

# scipy.io.wavfile.write('orig.wav', sr, y)
# y = scipy.signal.convolve(y, h, mode='same')
# scipy.io.wavfile.write('test.wav', sr, y)

# res, rem = scipy.signal.deconvolve(y, h)
# res /= np.max(np.abs(res))
# print(res)

plt.plot(y)
plt.plot(filtered)
plt.show()