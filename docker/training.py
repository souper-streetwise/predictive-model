from typing import Union, Dict
from sklearn.model_selection import cross_val_score
from skgarden import ExtraTreesQuantileRegressor
import warnings
from datetime import datetime
import pickle
import pandas as pd
from .utils import get_path, TQDM

def train_model(X: object, y: object,
    n_estimators: int = 1000,
    max_depth: Union[int, None] = None,
    min_samples_split: Union[int, float] = 2,
    min_samples_leaf: Union[int, float] = 3,
    min_weight_fraction_leaf: float = 0.,
    max_features: Union[int, float, str, None] = 'auto',
    max_leaf_nodes: Union[int, None] = None,
    criterion: str = 'mse',
    bootstrap: bool = True,
    save_model: bool = True,
    cv: int = 10,
    workers: int = -1,
    data_dir: str = 'data',
    model_name: str = 'soup_model',
    **kwargs) -> Dict[str, object]:

    # Round hyperparameters
    min_weight_fraction_leaf = round(min_weight_fraction_leaf, 2)
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
        min_weight_fraction_leaf = min_weight_fraction_leaf,
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
        with open(get_path(data_dir) / model_name, 'wb') as f:
            pickle.dump(model_data, f)

    return model_data

if __name__ == '__main__':
    pass
