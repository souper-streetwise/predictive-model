def train_model(X, y, n_iter: int = 100, cv: int = 10, data_dir: str = 'data', 
    workers: int = -1, save_log: bool = True, save_model: bool = True, 
    model_name: str = 'soup_model', **kwargs) -> dict:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    import warnings
    from model import pExtraTreesRegressor
    from utils import get_path, TQDM

    hyperparams = {
        'n_estimators': Integer(10, 10000),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'bootstrap': Categorical([True, False])
    }

    search = BayesSearchCV(
        estimator = pExtraTreesRegressor(criterion = 'mae', n_jobs = workers), 
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
    
    model = search.best_estimator_

    feat_dict = dict(list(zip(model.feature_importances_, X.columns)))
    feat_sorted = sorted(feat_dict, reverse = True)
    feat_imps = [(feat_dict[imp], imp) for imp in feat_sorted]

    model_data = {
        'model': model, 
        'score': -search.best_score_,
        'feat_imps': feat_imps
    }

    if save_model:
        import pickle
        with open(get_path(data_dir) / model_name, 'wb') as f:
            pickle.dump(model_data, f)

    if save_log:
        import pandas as pd
        results = pd.DataFrame(search.cv_results_)
        results_path = get_path(data_dir) / f'{model_name}_training_log.csv'
        results.to_csv(results_path, index = False)

    return model_data, results

if __name__ == '__main__':
    from utils import get_path, boolean
    from data import get_data
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--n_iter', type = int, default = 100)
    parser.add_argument('--cv', type = int, default = 10)
    parser.add_argument('--workers', type = int, default = -1)
    parser.add_argument('--data_dir', default = 'data')
    parser.add_argument('--save_log', type = boolean, default = True)
    parser.add_argument('--save_model', type = boolean, default = True)
    parser.add_argument('--model_name', default = 'soup_model')
    parser.add_argument('--include_month', type = boolean, default = True)
    parser.add_argument('--include_day_of_week', type = boolean, default=True)
    parser.add_argument('--include_day_of_month', type = boolean, default=True)
    args = vars(parser.parse_args())

    X, y = get_data(**args)
    model_data, results = train_model(X, y, **args)
    print(model_data)
