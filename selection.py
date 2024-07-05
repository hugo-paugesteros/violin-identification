import pandas as pd
import librosa
import numpy as np
import sklearn.neighbors, sklearn.preprocessing, sklearn.metrics, sklearn.pipeline, sklearn.svm
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from features import *
import tqdm
import matplotlib.pyplot as plt

SR          = 16000
FRAME_SIZE  = 512
HOP_SIZE    = FRAME_SIZE
N_COEFF     = FRAME_SIZE//2+1

df = pd.read_pickle('recordings.pkl')
df = df[df['type']=='scale']

data = []
for index, row in tqdm.tqdm(df.iterrows()):
    audio, _ = librosa.load(str(row['file']), sr=SR)
    # features = LTAS(audio, FRAME_SIZE, HOP_SIZE, N_COEFF)
    _, features = 10 * np.log10(scipy.signal.welch(audio, nperseg=FRAME_SIZE))
    dic = dict(zip(np.arange(len(features)), features))
    dic.update(player=row['player'], violin=row['violin'], type=row['type'])
    data.append(dic)
    # features = LTAS_third(audio, SR, FRAME_SIZE, HOP_SIZE)
    # N_COEFF = features.shape[1] 
    # features = MFCC(audio, SR, FRAME_SIZE, HOP_SIZE, N_COEFF)
    # features = LTCC(audio, FRAME_SIZE, HOP_SIZE, N_COEFF)
    # for i in range(features.shape[0]):
    #     dic = dict(zip(list(range(features.shape[1])), features[i].T))
    #     dic.update(player=row['player'], violin=row['violin'], type=row['type'])
    #     data.append(dic)

features_df = pd.DataFrame(data)
X = features_df.iloc[:, :N_COEFF]
y = features_df['violin']

selection = SelectKBest(f_classif, k='all')
selection.fit_transform(X, y)
scores = -np.log10(selection.pvalues_)
bins = selection.get_feature_names_out(input_features=np.array(range(N_COEFF))).astype(np.float32)
# freqs = bins / (FRAME_SIZE//2+1) * SR/2
print(bins)

fig, ax = plt.subplots()
ax.bar(np.arange(N_COEFF) / (FRAME_SIZE//2+1) * SR/2, scores, align='center', width=20)
ax.set_title('LTAS coefficients univariate score')
ax.set_xlabel('Feature number')
ax.set_ylabel(r'Univariate score ($-Log(p_{value})$)')
plt.show()
