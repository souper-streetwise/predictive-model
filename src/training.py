def train_model(X, y, n_iter: int = 100, cv: int = 10, data_dir: str = 'data', 
    workers: int = -1, eps: float = 1e-7) -> dict:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    import pickle
    import warnings
    import pandas as pd
    from modules import pExtraTreesRegressor, TQDM 
    from utils import get_path

    hyperparams = {
        'n_estimators': Integer(10, 5000),
        'max_depth': Integer(10, 1000),
        'min_samples_split': Real(eps, 1.0, prior = 'uniform'),
        'min_samples_leaf': Real(eps, 0.5, prior = 'uniform'),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'max_leaf_nodes': Integer(2, 10000),
        'min_impurity_decrease': Real(0.0, 1.0, prior = 'uniform'),
        'bootstrap': Categorical([True, False]),
        'max_samples': Real(eps, 1 - eps, prior = 'uniform')
    }

    estimator = pExtraTreesRegressor(criterion = 'mae', n_jobs = workers)

    with TQDM(total = n_iter, desc = 'Optimising ExtraTrees') as pbar:
        search = BayesSearchCV(
            estimator = estimator, 
            search_spaces = hyperparams, 
            n_iter = n_iter,
            scoring = 'neg_mean_absolute_error',
            cv = cv,
            n_jobs = workers,
        )

        with warnings.catch_warnings():
            message = 'The objective has been evaluated at this point before.'
            warnings.filterwarnings('ignore', message = message)
            search.fit(X, y, callback = pbar)
    
    model = search.best_estimator_

    feat_dict = dict(list(zip(model.feature_importances_, X.columns)))
    feat_sorted = sorted(feat_dict, reverse = True)
    feat_imps = [(feat_dict[imp], imp) for imp in feat_sorted]

    model_data = {
        'model': model, 
        'score': -search.best_score_,
        'feat_imps': feat_imps
    }

    results = pd.DataFrame(search.cv_results_)
    results_path = get_path(data_dir) / 'extra_trees_training_log.csv'
    results.to_csv(results_path, index = False)

    with open(get_path(data_dir) / 'extra_trees', 'wb') as f:
        pickle.dump(model_data, f)

    return model_data, results

if __name__ == '__main__':
    from utils import get_path
    from data import get_data
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--n_iter', type = int, default = 100)
    parser.add_argument('--cv', type = int, default = 10)
    parser.add_argument('--workers', type = int, default = -1)
    parser.add_argument('--data_dir', default = 'data')
    args = vars(parser.parse_args())

    model_data, results = train_model(*get_data(), **args)
    print(model_data)
