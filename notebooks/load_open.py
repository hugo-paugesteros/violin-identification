import numpy as np
import librosa
import os

SR = 10000

path = '/home/hugo/Thèse/segmentation/processed/'

violin_1 = []
violin_2 = []

for file in os.listdir(path):
    if file.endswith('.wav'):
        file = file[:-4]
        T = int(file[7:])
        y, sr = librosa.load(path + file + '.wav', sr=SR)
        if(T <= 4):
            violin_1.append(y)
        else:
            violin_2.append(y)

violin_1 = np.concatenate(violin_1)
violin_2 = np.concatenate(violin_2)
print(violin_1.shape)
print(violin_2.shape)

np.save(f'violin_1.npy', violin_1)
np.save(f'violin_2.npy', violin_2)