import glob
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import librosa

SR = 10000
FRAME_SIZE = 4096*8
# FRAME_SIZE = 32768*16
HOP_SIZE = FRAME_SIZE//2
N_MFCC = 30

print(f'Frame size : {np.round(FRAME_SIZE / SR * 100)} ms')

Pxxs = []
plt.figure()
for file in glob.glob(f'*.npy')[:]:
    y = np.load(file)

    # (f, Pxx) = scipy.signal.welch(y, fs=SR, nperseg=FRAME_SIZE, noverlap=HOP_SIZE, average='mean')
    # Pxx = np.log10(Pxx)
    # plt.plot(f, Pxx, label=file[:-4])
    
    Pxx = librosa.feature.mfcc(y=y, n_mfcc=N_MFCC, sr=SR, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    Pxx = np.mean(Pxx, axis=-1)
    plt.plot(Pxx, label=file[:-4])

    Pxxs.append(Pxx)

file = '../Data/SegmentationPerViolin/PLAYER6/Violin5.wav'
y, sr = librosa.load(file, sr=SR)
# (f, Pxx) = scipy.signal.welch(y, fs=SR, nperseg=FRAME_SIZE, noverlap=HOP_SIZE, average='mean')
# # Pxx = np.log10(Pxx)
# plt.plot(f, Pxx, label='test')

Pxx = librosa.feature.mfcc(y=y, n_mfcc=N_MFCC, sr=SR, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Pxx = np.mean(Pxx, axis=-1)
plt.plot(Pxx)

Pxxs = np.array(Pxxs)
corr = np.sum(Pxx * Pxxs, axis=-1)
print(corr)
print(np.argsort(corr)+1)

plt.legend()
# plt.xscale('log')
# plt.xlim([200, 5000])
plt.show()