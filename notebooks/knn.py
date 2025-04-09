import glob
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import librosa

SR = 10000

FRAME_SIZE = 4096
# FRAME_SIZE = 32768*4
HOP_SIZE = FRAME_SIZE
# N_MFCC = 30
def LTAS(y):
    Y = librosa.stft(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    Y = librosa.feature.melspectrogram(S=np.abs(Y)**2, n_mels=25)
    S = 10 * np.log10(np.abs(Y))[:25]
    return S.T

def LTAS_third(y):
    y = librosa.util.frame(y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE).T
    CENTER_FREQ = 1000
    freqs = CENTER_FREQ * np.power(2**(1/3), np.arange(-18, 7))
    # freqs = 7000 * np.power(2**(1/3), np.arange(-16, 1))
    # freqs = CENTER_FREQ * 2.0 ** (np.arange(-1, 9)-5)
    freqs_lower = freqs / 2**(1/6)
    freqs_upper = freqs * 2**(1/6)
    LTAS = np.empty((y.shape[0], freqs.shape[0]))
    for k in range(len(freqs)):
        lower = freqs_lower[k]
        upper = freqs_upper[k]
        # factor = ((sr / 2) / (freqs_upper/2**(1/6)))
        # print(np.round(len(y) / factor))
        # sd = scipy.signal.resample(y, np.round(len(y) / factor))
        sos = scipy.signal.butter(N=3, Wn=np.array([lower, upper])/(SR/2), btype='bandpass', analog=False,output='sos')
        band = scipy.signal.sosfiltfilt(sos, y)
        LTAS[:, k] = 10 * np.log10(np.sum(np.power(band,2), axis=-1))
    # print(LTAS)
    LTAS -= LTAS[:, 0:1]
    return LTAS[:, 1:]

def LTCC(y):
    Y = librosa.stft(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE).T
    S = 20 * np.log10(np.abs(Y) + 0.001)
    ltcc = np.fft.irfft(S)[:,:60]
    return ltcc

def MFCC(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    return mfcc.T

print(f'Frame size : {np.round(FRAME_SIZE / SR * 1000)} ms')

x = []
y = []
for i in range(1,14):
    print(f'Violin {i}')
    # audio = np.load(f'../Data/BachSei/{folder}.npy')
    audio = np.load(f'violin{i}.npy')
    audio /= np.max(audio)
    
    # features = LTAS(audio)
    # features = LTAS_third(audio)
    features = LTCC(audio)
    # features = MFCC(audio)

    x.append(features)
    y.append(np.ones(features.shape[0])*i)

x = np.concatenate(x)
y = np.concatenate(y)

x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
np.save('LTAS_third_x', x)
np.save('LTAS_third_y', y)

x = np.load('LTAS_third_x.npy')
y = np.load('LTAS_third_y.npy')

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1)

# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13, weights='uniform', p=1) # n_neighbors = k
knn.fit(x_train,y_train)
# prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(13,knn.score(x_train,y_train)))
print(" {} nn score: {} ".format(13,knn.score(x_test,y_test)))


# x_test, _ = librosa.load('/home/hugo/Thèse/Data/BilbaoViolins/Recordings_FreeCat/PLAYER22/player22 violin13.wav', sr=SR)
# x_test = LTAS_third(x_test)

# # x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
# y_test = np.ones(x_test.shape[0])*7
# print(y_test)
# print(" {} nn score: {} ".format(13,knn.score(x_test,y_test)))

y_pred = knn.predict(x_test)
# print(y_pred)
y_true = y_test

#confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
# print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred, labels=list(range(1,14)))

import seaborn as sns 
f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()