import matplotlib.pyplot as plt
import pandas as pd
import librosa
import numpy as np
import sklearn.neighbors, sklearn.preprocessing, sklearn.metrics, sklearn.pipeline, sklearn.svm, sklearn.feature_selection, sklearn.neural_network, sklearn.model_selection
import tqdm

from src.features import *

base_config = {
    # 'frame_size': [2**n for n in range(11, 16)],
    'frame_size': [2048],
    'hop_ratio': [1],
    # 'n_coeff': [10*i for i in range(4,10, 2)],
    'n_coeff': [100],
    'sr': [10000],
    'size': [60],
    'feature': ['MFCC_welch']
}

configs = [
{
    'clf': [sklearn.neighbors.KNeighborsClassifier()],
    # 'n_neighbors': [10*i+1 for i in range(4,10, 2)],
    'n_neighbors': [41],
    'p': [1],
    'weights': ['distance'], 
},
# {
#     # 'clf': [sklearn.svm.SVC(decision_function_shape='ovr')]
# },
# {
#     # 'clf': [sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(30,30), max_iter=2000)]
# }
]

configs = [config | base_config for config in configs]

df = pd.read_pickle('data/processed/dataset_cnsm.pkl')
df = df[(df.violin.isin(['A', 'B', 'C']))]

train_cdt = df.session == 1
test_cdt = ~train_cdt

print(train_cdt.value_counts())

df = df[train_cdt | test_cdt]

def train(config):
    # Features
    data = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        offset = row['start']
        duration = row['end'] - offset
        y, _    = librosa.load(str(row['file']), sr=config['sr'], offset=offset, duration=duration)
        # scipy.io.wavfile.write(
        #     f'data/test/{row['violin']}{row['sample']}{row['condition']}.wav',
        #     config['sr'],
        #     y
        # )
        # for i, audio in  enumerate(np.lib.stride_tricks.sliding_window_view(y, window_shape=10*config['sr'])[::10*config['sr']]):
        # for audio in np.split(y, np.arange(config['sr']*config['size'], len(y), config['sr']*config['size'])):
        for audio in [y]:

            features = audio
            for step in pipes[config['feature']]:
                features = step(features, **config)

            dic = row.to_dict()
            dic.update(
                features=features,
                audio=audio
            )
            data.append(dic)

    features_df = pd.DataFrame(data)
    features_df.to_pickle('data/processed/features_cnsm.pkl')

    # Test / Train
    x_train = np.vstack(features_df[train_cdt.to_numpy()].features)
    y_train = features_df[train_cdt.to_numpy()].violin.to_numpy()
    x_test = np.vstack(features_df[test_cdt.to_numpy()].features)
    y_test = features_df[test_cdt.to_numpy()].violin.to_numpy()

    # Train
    # nca = sklearn.neighbors.NeighborhoodComponentsAnalysis(max_iter=1, verbose=1)
    estimator = config['clf']
    estimator.set_params(**config)
    pipeline = sklearn.pipeline.Pipeline([
        # ("anova", sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif, percentile=50)),
        ('scaler', sklearn.preprocessing.MaxAbsScaler()),
        # ('nca', nca),
        ('knn', estimator),
        # ('svm', svm),
        # ('MLP', sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(30,30), max_iter=2000)),
    ])
    pipeline.fit(x_train, y_train)

    print(f'Train score : {pipeline.score(x_train, y_train)}')
    print(f'Test score : {pipeline.score(x_test, y_test)}')

    # Save
    # y_pred = pipeline.predict(x_test)
    # for i, (_, row) in enumerate(features_df.query(test_cdt).iterrows()):
    #     folder = 'good' if y_pred[i] == y_test[i] else 'bad'
    #     scipy.io.wavfile.write(
    #         f'data/{folder}/{str(row['file']).replace('/', '-')}{i}.wav',
    #         config['sr'],
    #         row['audio']
    #     )

configs = sklearn.model_selection.ParameterGrid(configs)

for config in configs:
    config['hop_size'] = config['frame_size'] // config['hop_ratio']
    print(config)
    train(config)
    print('---------------------------')

# # Plots
# disp = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(
#     pipeline,
#     x_test, y_test,
#     labels=list(set(y_test)),
#     display_labels=list(set(y_test)),
#     normalize='pred',
#     cmap='Blues',
#     colorbar=False
# )
# plt.xticks(rotation=45)
# plt.gcf().set_size_inches(10, 10)
# plt.savefig('figures/confusion.png', bbox_inches='tight', dpi=400)
# plt.show()