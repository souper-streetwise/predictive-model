def train_forest(X, y, cv: int = 10, n_iter: int = 1000, 
    data_dir: str = 'data') -> dict:
    ''' Train a random forest. '''
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV
    from models import pExtraTreesRegressor
    import pickle
    from utils import get_path
    import numpy as np

    hyperparams = {
        'n_estimators': list(range(100, 2000, 100)),
        'max_features': ['auto', 'sqrt'],
        'max_depth': list(range(10, 100, 10)) + [None],
        'min_samples_split': list(range(2, 10)),
        'min_samples_leaf': list(range(1, 5)),
        'min_weight_fraction_leaf': list(np.arange(0, 0.5, 0.05)),
        'min_impurity_decrease': list(np.arange(0, 0.5, 0.05)),
        'bootstrap': [True, False]
    }

    rnd_search = RandomizedSearchCV(
        estimator = pExtraTreesRegressor(), 
        param_distributions = hyperparams, 
        n_iter = n_iter, 
        cv = cv, 
        verbose = 1, 
        n_jobs = -1
    )

    rnd_search.fit(X, y)
    model = rnd_search.best_estimator_

    feat_dict = dict(list(zip(model.feature_importances_, X.columns)))
    feat_sorted = sorted(feat_dict, reverse = True)
    feat_imps = [(feat_dict[imp], imp) for imp in feat_sorted]

    scores = cross_val_score(model, X, y, cv = cv, n_jobs = -1, verbose = 1)

    output = {
        'model': model, 
        'scores': np.abs(scores), 
        'params': rnd_search.best_params_,
        'feat_imps': feat_imps
    }

    with open(get_path(data_dir) / 'extra_trees', 'wb') as f:
        pickle.dump(output, f)

    return output

if __name__ == '__main__':
    from utils import get_path
    from data import get_data

    with open(get_path('darksky_key.txt'), 'r') as file_in:
        KEY = file_in.read().rstrip()

    X, y = get_data()
    model_data = train_forest(X, y)
    print(model_data['scores'])
