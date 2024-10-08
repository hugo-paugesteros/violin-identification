import sklearn.neighbors, sklearn.pipeline, sklearn.preprocessing, sklearn.svm, sklearn.model_selection
import pandas as pd

from ..dataset.dataset import Dataset
from ..pipeline import Pipeline

## GRIDS
dataset_grid = {
    'dataset__frame_size': [2048],
    'dataset__hop_ratio': [1],
    # 'dataset__n_coeff': [10*i for i in range(5,6)],
    'dataset__sr': [10000],
    'dataset__features': ['LTAS_welch_db', 'MFCC_welch', 'LTCC_stft'],
    'dataset__n_coeff': [100],
    # 'dataset__sample_duration': [10, 20, 30],
}

clf_grids = [{
    'clf': [sklearn.neighbors.KNeighborsClassifier()],
    'clf__n_neighbors': [41],
    'clf__p': [1],
    'clf__weights': ['distance'], 
},
# {
#     'clf': [sklearn.svm.SVC(decision_function_shape='ovr')],
#     'clf__decision_function_shape': ['ovr'],
# }
]

grid = [clf_grid | dataset_grid for clf_grid in clf_grids]

## Pipeline
pipeline = Pipeline([
    ('dataset', Dataset()),
    ('scaler', sklearn.preprocessing.MaxAbsScaler()),
    ('clf', sklearn.neighbors.KNeighborsClassifier())
])

## Dataset
df = pd.read_pickle('data/processed/dataset_cnsm.pkl')

## Grid search
grid_search = sklearn.model_selection.GridSearchCV(pipeline, grid, verbose=0, cv=5, n_jobs=-1)
grid_search.fit(df, df['violin'])

## Save results
df = pd.DataFrame(grid_search.cv_results_)
df.to_pickle('models/results.pkl')
df.to_csv('models/results.csv', float_format='{:.2f}'.format)