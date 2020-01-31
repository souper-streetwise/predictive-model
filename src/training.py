def train_model(X, y, cv: int = 10, n_iter: int = 10, data_dir: str = 'data'):
    ''' Train a random forest. '''
    from sklearn.model_selection import cross_val_score
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    from models import pExtraTreesRegressor
    import pickle
    from utils import get_path
    import numpy as np

    hyperparams = {
        'n_estimators': Integer(100, 2000),
        'max_features': Categorical(['auto', 'sqrt']),
        'max_depth': Integer(10, 1000),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
        'min_weight_fraction_leaf': Real(0, 0.5, prior='uniform'),
        'min_impurity_decrease': Real(0, 0.5, prior='uniform'),
        'bootstrap': Categorical([True, False])
    }

    bayes_search = BayesSearchCV(
        estimator = pExtraTreesRegressor(criterion = 'mae'), 
        search_spaces = hyperparams, 
        n_iter = n_iter,
        scoring = 'neg_mean_absolute_error',
        cv = cv,
        n_jobs = -1
    )

    bayes_search.fit(X, y)
    model = bayes_search.best_estimator_

    print('Search finished; the best parameters found were:')
    print(bayes_search.best_params_)

    feat_dict = dict(list(zip(model.feature_importances_, X.columns)))
    feat_sorted = sorted(feat_dict, reverse = True)
    feat_imps = [(feat_dict[imp], imp) for imp in feat_sorted]

    scores = cross_val_score(model, X, y, cv = cv, n_jobs = -1, 
        scoring = 'neg_mean_absolute_error')

    output = {
        'model': model, 
        'scores': np.abs(scores), 
        'params': bayes_search.best_params_,
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
    model_data = train_model(X, y)
    print(model_data['scores'])
