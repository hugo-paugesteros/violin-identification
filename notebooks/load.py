import numpy as np
import librosa
import os

SR = 10000

for i in range(1,14):
    print(f'Violin {i}')
    conc = []
    for j in range(1,27):
        # if j == 3:
        #     continue
        print('\t' + f'Player {j}')
        file = f'../Data/SegmentationPerViolin/PLAYER{j}/Violin{i}.wav'
        if(os.path.exists(file)):
            y, sr = librosa.load(file, sr=SR)
            conc.append(y)

    conc = np.concatenate(conc)
    np.save(f'violin{i}.npy', conc)