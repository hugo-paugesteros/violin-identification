import sklearn.neighbors, sklearn.pipeline, sklearn.preprocessing, sklearn.svm, sklearn.model_selection, sklearn.neural_network
import pandas as pd
import numpy as np

from ..dataset.dataset import Dataset
from ..pipeline import Pipeline
from ..GaussianMixtureClassifier import GaussianMixtureClassifier

## GRIDS
dataset_grid = {
    'dataset__frame_size': [2048],
    'dataset__hop_ratio': [1],
    # 'dataset__n_coeff': [10*i for i in range(2,10)],
    'dataset__sr': [10000],
    'dataset__features': ['MFCC_welch'],
    'dataset__n_coeff': [100],
    'dataset__sample_duration': [30000],
}

clf_grids = [
{
    'clf': [sklearn.neighbors.KNeighborsClassifier()],
    'clf__n_neighbors': [41],
    'clf__p': [1],
    'clf__weights': ['distance'], 
},
# {
#     'clf': [sklearn.svm.SVC(decision_function_shape='ovr')],
#     'clf__decision_function_shape': ['ovr'],
# },
# {
#     'clf': [sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(30,30), max_iter=2000)]
# },
# {
#     'clf': [GaussianMixtureClassifier()],
#     'clf__n_components': [3],
# },
]

grid = [clf_grid | dataset_grid for clf_grid in clf_grids]

## Pipeline
pipeline = Pipeline([
    ('dataset', Dataset()),
    ('scaler', sklearn.preprocessing.MaxAbsScaler()),
    ('clf', sklearn.neighbors.KNeighborsClassifier())
],
# memory='tmp/'
)

## Dataset
df = pd.read_pickle('data/processed/dataset_cnsm.pkl')
# df = df[pd.Series(np.random.rand(len(df)) < 0.2)]
df = df[(df.violin.isin(['A', 'B', 'C']))]
# df = df[df.type.isin(['scale', 'free'])]

le = sklearn.preprocessing.LabelEncoder()
df['violin'] = le.fit_transform(df['violin'])
df['player'] = le.fit_transform(df['player'])

## Grid search
grid_search = sklearn.model_selection.GridSearchCV(pipeline, grid, verbose=0, cv=5, n_jobs=-1)
grid_search.fit(df, df['player'])

## Save results
df = pd.DataFrame(grid_search.cv_results_)
df.to_pickle('models/results.pkl')
df.to_csv('models/results.csv', float_format='{:.2f}'.format)