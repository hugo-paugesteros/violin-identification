import matplotlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
import scipy.spatial
import pandas as pd
import scipy.cluster._hierarchy

from src.features import *

SR          = 10000
FRAME_SIZE  = 2048
HOP_SIZE    = FRAME_SIZE
N_COEFF     = 50
FEATURE     = 'MFCC_welch'

data = pd.read_csv('/home/hugo/Thèse/Data/Synthétiques/SelectionSZViolons_Test3_Cplx/Z-210 01SpectraPlus R R_Cplx.tsv', sep='\t')

# print(data['real'].to_numpy() + 1j * data['complex'].to_numpy())

files = sorted(glob.glob(f'/home/hugo/Thèse/Data/Synthétiques/sounds/*.wav'))
print(files)
mfccs = []
for file in files:
    y, sr = librosa.load(file, sr=SR)
    features = y
    for step in pipes[FEATURE]:
        features = step(features, **{'sr': SR, 'frame_size': FRAME_SIZE, 'hop_size': HOP_SIZE, 'n_coeff': N_COEFF})
    mfccs.append(features)

mfccs = np.array(mfccs)

names = [file[-21:-16] for file in files]
distance_matrix = scipy.spatial.distance_matrix(mfccs, mfccs, p=1)
Z = scipy.cluster.hierarchy.linkage(distance_matrix, 'single')

fig, ax = plt.subplots()
# ax.imshow(distance_matrix, origin='upper')
dn = scipy.cluster.hierarchy.dendrogram(Z, labels=names)
# ax.set_xticks(range(len(files)), [file[-21:-16] for file in files], rotation=45)
# ax.set_yticks(range(len(files)), [file[-21:-16] for file in files])
plt.show()