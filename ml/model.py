from typing import Union, Dict
from sklearn.model_selection import cross_val_score
from skgarden import ExtraTreesQuantileRegressor
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load, dump

def load_model_data(model_name: str = 'soup_model', 
    data_dir: Union[str, Path] = '.data') -> Dict[str, object]:
    ''' Load a saved model with associated scores and feature importances. 
    
    INPUT
        model_name: str
            The file name of the pickled model
        data_dir: Union[str, pathlib.Path]
            The absolute path to the directory containing the model

    OUTPUT
        A dictionary containing:
            model: skgarden.ExtraTreesQuantileRegressor
                The machine learning model
            score: float
                The mean absolute error from a cross validation
            feat_imps: List[float, str]
                A list of the features with their feature importance
                score, sorted in descending order
    '''
    model_path = Path(data_dir) / model_name
    if model_path.exists():
        return load(model_path)
    else:
        raise FileNotFoundError(f'The model {model_path} was not found.')

def train_model(
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray],
    n_estimators: int = 1000,
    max_depth: Union[int, None] = None,
    min_samples_split: Union[int, float] = 2,
    min_samples_leaf: Union[int, float] = 3,
    max_features: Union[int, float, str, None] = 'auto',
    max_leaf_nodes: Union[int, None] = None,
    criterion: str = 'mse',
    bootstrap: bool = True,
    save_model: bool = True,
    cv: int = 10,
    workers: int = -1,
    data_dir: Union[str, Path] = '.data',
    model_name: str = 'soup_model',
    **kwargs) -> Dict[str, object]:
    ''' Train an ExtraTreesQuantileRegressor model from scratch.

    INPUT
        X: Union[pandas.DataFrame, numpy.ndarray]
            The feature matrix
        y: Union[pandas.Series, numpy.ndarray]
            The target values
        n_estimators: int = 1000
            The number of trees in the forest
        max_depth: Union[int, None] = None
            The maximum depth of each tree, with None meaning no limit
        min_samples_split: Union[int, float] = 2
            The minimum number of samples required to perform a split
            through a tree. If a float is given then this is the
            proportion of the number of rows in X
        min_samples_leaf: Union[int, float] = 3
            The minimum number of samples required in each leaf. This
            is an important parameter in the case of a Quantile
            Regression Forest, as the distribution of samples in a
            given leaf determines the quantile prediction. Normally
            this defaults to 1, but because it is more important to
            us it is set to 3. If a float is given then this is the
            proportion of the number of rows in X
        max_features: Union[int, float, str, None] = 'auto'
            The number of features to consider when performing a split.
            If a float is given then this is the proportion of the number
            of rows in X. If None is given then all features are used.
            If a string is given then this can be either
                'auto' or 'sqrt': This uses sqrt(num_features) features
                'log2: This uses log2(num_features) features
        max_leaf_nodes: Union[int, None] = None
            How many leaf nodes are permitted. If None is given then
            there is no upper bound.
        criterion: str = 'mse'
            The function used to determine how good a split is. Can
            be 'mse' for mean squared error, or 'mae' for mean absolute
            error.
        bootstrap: bool = True
            Whether to train each tree on a bootstrapped sample of
            the data
        save_model: bool = True
            Whether to save the model after training. It will be saved
            to <data_dir> / <model_name>
        cv: int = 10
            The number of cross-validation folds when computing the
            score of the model
        workers: int = -1
            The number of CPU cores to use when training the model. If
            -1 is given then all CPU cores will be used
        data_dir: Union[str, pathlib.Path] = '.data'
            The path where the model should be stored
        model_name: str = 'soup_model'
            The filename of the model

    OUTPUT
        A dictionary containing:
            model: skgarden.ExtraTreesQuantileRegressor
                The trained machine learning model
            score: float
                The mean absolute error from a <cv>-fold cross validation
            feat_imps: List[float, str]
                A list of the features with their feature importance
                score, sorted in descending order
    '''
    # Round hyperparameters
    if isinstance(max_depth, int) and max_depth > 5000: 
        max_depth = None
    if isinstance(min_samples_split, float):
        min_samples_split = round(min_samples_split, 2)
        if min_samples_split == 0.0: min_samples_split = 2
    if isinstance(min_samples_leaf, float):
        min_samples_leaf = round(min_samples_leaf, 2)
        if min_samples_leaf == 0.0: min_samples_leaf = 3
    if isinstance(max_features, float):
        max_features = round(max_features, 2)
        if max_features == 0.0: max_features = 1
    if isinstance(max_leaf_nodes, int) and max_leaf_nodes > 5000:
        max_leaf_nodes = None

    model = ExtraTreesQuantileRegressor(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        max_leaf_nodes = max_leaf_nodes,
        criterion = criterion,
        bootstrap = bootstrap,
        n_jobs = workers,
    )

    score = -cross_val_score(model, X, y, 
        scoring = 'neg_mean_absolute_error', 
        cv = cv, 
        n_jobs = workers, 
        verbose = 1
    ).mean()

    model = model.fit(X, y)
    
    feat_dict = dict(list(zip(model.feature_importances_, X.columns)))
    feat_sorted = sorted(feat_dict, reverse = True)
    feat_imps = [(feat_dict[imp], imp) for imp in feat_sorted]

    model_data = {'model': model, 'score': score, 'feat_imps': feat_imps}

    if save_model:
        if not Path(data_dir).is_dir():
            Path(data_dir).mkdir()
        dump(model_data, Path(data_dir) / model_name)

    return model_data
