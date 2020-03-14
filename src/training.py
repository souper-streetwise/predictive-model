from typing import Union, Dict

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

    from sklearn.model_selection import cross_val_score
    from skgarden import ExtraTreesQuantileRegressor
    from utils import get_path

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
        import pickle
        with open(get_path(data_dir) / model_name, 'wb') as f:
            pickle.dump(model_data, f)

    return model_data

def get_best_params(X: object, y: object,
    n_iter: int = 100, 
    cv: int = 10, 
    eps: float = 1e-7,
    data_dir: str = 'data', 
    workers: int = -1, 
    save_log: bool = True,
    model_name: Union[str, None] = None,
    **kwargs) -> Dict[str, Union[float, int, bool, str]]:

    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    from skgarden import ExtraTreesQuantileRegressor
    import warnings
    from datetime import datetime
    from utils import get_path, TQDM

    hyperparams = {
        'n_estimators': Integer(100, 5000),
        'max_depth': Integer(2, 10000),
        'min_samples_split': Integer(2, 100),
        'min_samples_leaf': Integer(3, 100),
        'min_weight_fraction_leaf': Real(0., 0.5, prior = 'uniform'),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'max_leaf_nodes': Integer(10, 10000),
        'criterion': Categorical(['mse', 'mae']),
        'bootstrap': Categorical([True, False])
    }
    
    estimator = ExtraTreesQuantileRegressor(n_jobs = workers)
    search = BayesSearchCV(
        estimator = estimator, 
        search_spaces = hyperparams, 
        n_iter = n_iter,
        scoring = 'neg_mean_absolute_error',
        cv = cv,
        n_jobs = workers
    )

    with TQDM(total = n_iter, desc = 'Optimising model') as pbar:
        with warnings.catch_warnings():
            message = 'The objective has been evaluated at this point before.'
            warnings.filterwarnings('ignore', message = message)
            search.fit(X, y, callback = pbar)

    if save_log:
        import pandas as pd
        log_id = f'{model_name}_' if model_name is not None else ''
        log_id += datetime.now().strftime('%Y%m%dT%H%M%S')
        log_path = get_path(data_dir) / f'log_{log_id}.csv'
        pd.DataFrame(search.cv_results_).to_csv(log_path, index = False)
    
    return search.best_params_

if __name__ == '__main__':
    from utils import get_path, boolean
    from data import get_data
    from argparse import ArgumentParser

    def int_float(input: str):
        num = float(input)
        return num if num % 1 else int(num)

    parser = ArgumentParser()
    parser.add_argument('--n_iter', type = int, default = 100)
    parser.add_argument('--cv', type = int, default = 10)
    parser.add_argument('--workers', type = int, default = -1)
    parser.add_argument('--data_dir', default = 'data')
    parser.add_argument('--save_log', type = boolean, default = True)
    parser.add_argument('--save_model', type = boolean, default = True)
    parser.add_argument('--model_name', default = 'soup_model')
    parser.add_argument('--include_month', type = boolean, default = True)
    parser.add_argument('--include_day_of_week', type = boolean, 
        default = True)
    parser.add_argument('--include_day_of_month', type = boolean, 
        default = True)
    parser.add_argument('--optimise', type = boolean, default = False)
    parser.add_argument('--n_estimators', type = int, default = 1000)
    parser.add_argument('--max_depth', type = int, default = None)
    parser.add_argument('--min_samples_split', type = int_float, default = 2)
    parser.add_argument('--min_samples_leaf', type = int_float, default = 1)
    parser.add_argument('--min_weight_fraction_leaf', type = float, 
        default = 0.)
    parser.add_argument('--max_features', type = str, default = 'auto')
    parser.add_argument('--max_leaf_nodes', type = int, default = None)
    parser.add_argument('--bootstrap', type = boolean, default = True)
    parser.add_argument('--criterion', type = str, default = 'mse')
    args = vars(parser.parse_args())

    X, y = get_data(**args)
    if args['optimise']: args = {**args, **get_best_params(X, y, **args)}
    model_data = train_model(X, y, **args)
    print(model_data)
