import glob
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import librosa

SR = 10000

FRAME_SIZE = 2048
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

x = np.load('LTAS_third_x.npy')
y = np.load('LTAS_third_y.npy')

# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13, weights='uniform', p=1) # n_neighbors = k
knn.fit(x,y)
# prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(13,knn.score(x,y)))

x_test = []
y_test = []
# for i in range(1,14):
for i in range(1,25):
    print(f'Player {i}')
    # file = [f'../Data/SegmentationPerViolin/PLAYER3/Violin{i}.wav']
    file = glob.glob(f'../Data/BilbaoViolins/Recordings_FreeCat/PLAYER{i}/cut_player{i}*violin*.wav')
    if not file:
        continue
    # audio = np.load(f'../Data/BachSei/{folder}.npy')
    audio, _ =librosa.load(file[0], sr=SR)
    audio /= np.max(audio)

    
    # features = LTAS(audio)
    # features = LTAS_third(audio)
    features = LTCC(audio)
    # features = MFCC(audio)

    x_test.append(features)
    
    if file[0][-6].isdigit():
        vl = int(file[0][-6:-4])
    else:
        vl = int(file[0][-5])
    y_test.append(np.ones(features.shape[0])*vl)
    # y_test.append(np.ones(features.shape[0])*i)
    # y_test.append(np.ones(features.shape[0])*np.random.randint(1,14, features.shape[0]))

x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

print(" {} nn score: {} ".format(13,knn.score(x_test,y_test)))
y_pred = knn.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=list(range(1,14)))

import seaborn as sns 
f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()