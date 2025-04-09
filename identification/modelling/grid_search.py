import sklearn.neighbors, sklearn.pipeline, sklearn.preprocessing, sklearn.svm, sklearn.model_selection, sklearn.neural_network
import pandas as pd
import numpy as np
import tqdm
import librosa

from ..dataset.dataset import get_dataset
from ..GaussianMixtureClassifier import GaussianMixtureClassifier

## GRIDS
dataset_grid = {
    "frame_size": [2048],
    "hop_ratio": [1],
    # 'n_coeff': [10*i for i in range(4,10)],
    "n_coeff": [60],
    "sr": [10000],
    # 'sr': [5000*i for i in range(1, 10)],
    "feature": ["LTAS_welch", "LTCC_welch", "MFCC_welch"],
    # 'feature': ['MFCC_welch'],
    "sample_duration": [30],
}

clf_grid = [
    {
        "clf": [sklearn.neighbors.KNeighborsClassifier()],
        "clf__n_neighbors": [21],
        "clf__p": [1],
        "clf__weights": ["distance"],
    },
    {
        "clf": [sklearn.svm.SVC()],
        "clf__decision_function_shape": ["ovr"],
        # "clf__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "clf__kernel": ["linear"],
    },
    {
        "clf": [sklearn.neural_network.MLPClassifier()],
        "clf__hidden_layer_sizes": [(10,)],
        "clf__max_iter": [1000],
        "clf__solver": ["lbfgs"],
    },
    # {
    #     'clf': [GaussianMixtureClassifier()],
    #     'clf__n_components': [3],
    # },
]

## Dataset
# df = pd.read_pickle("data/processed/dataset_bilbao.pkl")
# df = df[df.type.isin(["free", "scale"]) & (df.player != 15)]

df = pd.read_pickle("data/processed/dataset_cnsm.pkl")
df = df[(df.violin.isin(["A", "B", "C"])) & (df.extract != "?")]
df = df.replace(["A", "B", "C"], ["1", "2", "3"])
# df = df[df.condition == 'aveugle']
# df = df.iloc[:50]


def train(config, features):
    estimator = config["clf"]
    print(estimator)
    print(config)
    pipeline = sklearn.pipeline.Pipeline(
        [
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("clf", estimator),
        ]
    )
    pipeline.set_params(**config)

    X = np.vstack(features.features)
    y = features.violin.to_numpy()

    # Split train / test
    # extracts = ["free", "gamme", "bach", "mozart", "tchai", "sibelius", "glazounov"]
    # train, test = sklearn.model_selection.train_test_split(features, test_size=0.2)
    # splits = [
    #     (train[train.extract == extract].index, test.index) for extract in extracts
    # ]

    splits = []
    for _ in range(10):
        train, test = sklearn.model_selection.train_test_split(features, test_size=0.2)
        splits.append((train.index, test.index))
    # print(splits)

    # Cross validation
    # skf = sklearn.model_selection.StratifiedKFold(shuffle=True, n_splits=5)
    # splits = skf.split(X, y)

    wrong = pd.DataFrame()
    res = {}
    for i, (train_index, test_index) in enumerate(splits):
        # i = extracts[i]
        # Test / Train
        x_train = X[train_index]
        y_train = y[train_index]
        x_test = X[test_index]
        y_test = y[test_index]

        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        # y_score = pipeline.predict_proba(x_test)
        precision, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(
            y_test, y_pred, average="macro"
        )
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        res[f"precision{i}"] = precision
        res[f"recall{i}"] = recall
        res[f"fscore{i}"] = fscore
        res[f"accuracy{i}"] = acc
        # res[f'roc{i}'] = sklearn.metrics.roc_auc_score(y_test, y_score, multi_class='ovr')

        # wrong.append(features.iloc[test_index].iloc[y_pred != y_test])
        wrong = pd.concat([wrong, features.iloc[test_index].iloc[y_pred != y_test]])

    return res, wrong


## GridSearch
dataset_configs = sklearn.model_selection.ParameterGrid(dataset_grid)
clf_configs = sklearn.model_selection.ParameterGrid(clf_grid)
results = []


from multiprocessing import Pool
from functools import partial

# Compute each dataset in a pool
with Pool() as pool:
    features = pool.map(partial(get_dataset, df=df), dataset_configs)
pool.close()
pool.join()

# For each dataset, compute all classifiers in a pool
for i, dataset_config in enumerate(dataset_configs):
    feature = features[i]
    with Pool() as pool2:
        rows, wrong = zip(*pool2.map(partial(train, features=feature), clf_configs))

    # For each classifier, add results
    for j, row in enumerate(rows):
        row = dataset_config | clf_configs[j] | row
        results.append(row)

    wrong = pd.concat(wrong)

results = pd.DataFrame(results)
print(results)
results.to_pickle("models/results.pkl")
results.to_csv("models/results.csv", float_format="{:.2f}".format)

wrong = wrong.drop("features", axis=1)
wrong = wrong.value_counts()
wrong.to_pickle("models/wrong.pkl")
