import matplotlib
import pandas as pd
import librosa
import numpy as np
import sklearn.neighbors, sklearn.preprocessing, sklearn.metrics, sklearn.pipeline, sklearn.svm, sklearn.feature_selection, sklearn.neural_network, sklearn.model_selection
from features import *
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import tqdm

matplotlib.rcParams["savefig.directory"] = ""

SR          = 10000
FRAME_SIZE  = 1024*2
HOP_SIZE    = FRAME_SIZE
N_COEFF     = 50
SIZE        = 120 * SR
# N_COEFF     = FRAME_SIZE//2+1

meter = pyln.Meter(SR)

players = [3,4,5,6,7,8,11,12,14,16,17,18,19,21,22,24]

df = pd.read_pickle('recordings.pkl')
# df = df[df['player'].isin(players)]
# df = df[df['type'] == 'scale']
# df = df[~df['player'].isin([99])]
# df = df[df['violin'].isin([1,4,5,9,11,13])]

data = []
for index, row in tqdm.tqdm(df.iterrows()):
    y, _ = librosa.load(str(row['file']), sr=SR)
    for audio in np.split(y, np.arange(SIZE, len(y), SIZE)):
        if len(audio) < 10*SR:
            print('passed')
            pass
        # loudness = meter.integrated_loudness(audio)
        # audio = pyln.normalize.loudness(audio, loudness, -20.0)
        # features = LTAS(audio, FRAME_SIZE, HOP_SIZE)
        # features = np.mean(LTAS_third(audio, SR, FRAME_SIZE, HOP_SIZE), axis=0)
        # _, features = 10 * np.log10(scipy.signal.welch(audio, nperseg=FRAME_SIZE))
        # features = np.mean(LTCC(audio, FRAME_SIZE, HOP_SIZE, N_COEFF), axis=0)
        features = np.mean(MFCC(audio, SR, FRAME_SIZE, HOP_SIZE, N_COEFF), axis=0)
        # dic = dict(zip(list(range(features.shape[1])), 10 * np.log10(features.mean(axis=0))))
        dic = dict(zip(np.arange(len(features)), features))
        dic.update(player=row['player'], violin=row['violin'], type=row['type'])
        data.append(dic)
        # features = LTAS_third(audio, SR, FRAME_SIZE, HOP_SIZE)
        N_COEFF = features.shape[-1]
        # features = MFCC(audio, SR, FRAME_SIZE, HOP_SIZE, N_COEFF)
        # features = MERT(audio)
        # dic = dict(zip(list(range(features.shape[0])), features.T))
        # dic.update(player=row['player'], violin=row['violin'], type=row['type'])
        # data.append(dic)
        # for i in range(features.shape[0]):
        #     dic = dict(zip(list(range(features.shape[1])), features[i].T))
        #     dic.update(player=row['player'], violin=row['violin'], type=row['type'])
        #     data.append(dic)

features_df = pd.DataFrame(data)

# train_cdt = (features_df['player'].isin(list(range(1,10))))
# train_cdt = (features_df['player'].isin(list(range(1,20))))
# train_cdt = (features_df['type'] == 'scale')
# train_cdt = (features_df['type'] == 'scale') & (features_df['player'].any() not in [3, 4])

train_cdt = (features_df['player'].isin(list(range(1,10))))
x_train = features_df[train_cdt].iloc[:, :N_COEFF]
y_train = features_df[train_cdt]['violin']

test_cdt = ~train_cdt
# test_cdt = (features_df['type'] == 'free')
x_test = features_df[test_cdt].iloc[:, :N_COEFF]
y_test = features_df[test_cdt]['violin']

print(x_train.shape)
print(x_test.shape)
nca = sklearn.neighbors.NeighborhoodComponentsAnalysis(max_iter=1, verbose=1)
# knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 40, weights='distance', p=1)
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 13, weights='distance', p=1)
svm = sklearn.svm.SVC(decision_function_shape='ovr')
pipeline = sklearn.pipeline.Pipeline([
    # ("anova", sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif, percentile=50)),
    ('scaler', sklearn.preprocessing.MaxAbsScaler()),
    # ('nca', nca),
    ('knn', knn),
    # ('svm', svm),
    # ('MLP', sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(30,30), max_iter=2000)),
])
pipeline.fit(x_train, y_train)
# cv = sklearn.model_selection.cross_validate(pipeline, x_train, y_train)
# print(cv['test_score'].mean())
print(f'Train score : {pipeline.score(x_train, y_train)}')
print(f'Test score : {pipeline.score(x_test, y_test)}')

# print(y_test == pipeline.predict(x_test))
# Plots
from sklearn.metrics import confusion_matrix, classification_report
# print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_test, pipeline.predict(x_test), labels=list(set(y_train)))
disp = sklearn.metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm/np.sum(cm), 
    display_labels=list(set(y_train)),
)
disp.plot(cmap='Blues', values_format='.1%', colorbar=False)
plt.xticks(rotation=45)
plt.gcf().set_size_inches(10, 10)
plt.savefig('figures/confusion.png', bbox_inches='tight', dpi=400)
plt.show()