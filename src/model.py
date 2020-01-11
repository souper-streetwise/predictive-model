def train_forest(X, y, cv: int = 10, n_iter: int = 100, 
    model_fname: str = 'random_forest', data_dir: str = 'data') -> dict:
    ''' Train a random forest. '''
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import pickle
    from utils import get_path

    hyperparams = {
        'n_estimators': list(range(100, 2000, 100)),
        'max_features': ['auto', 'sqrt'],
        'max_depth': list(range(10, 100, 10)) + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rnd_search = RandomizedSearchCV(
        estimator = RandomForestRegressor(), 
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

    with open(get_path(data_dir) / model_fname, 'wb') as f:
        pickle.dump(output, f)

    return output

def train_gaussian_process(X, y, cv: int = 10, data_dir: str = 'data',
    model_fname: str = 'gaussian_process'):
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.gaussian_process import GaussianProcessRegressor, kernels
    import numpy as np
    import pickle
    from utils import get_path

    k = kernels.RationalQuadratic()
    model = GaussianProcessRegressor(kernel = k, n_restarts_optimizer = 100)

    X_train, X_val, y_train, y_val = train_test_split(X, y)
    scores = cross_val_score(model, X, y, cv = cv, n_jobs = -1, verbose = 1)

    output = {'model': model.fit(X, y), 'scores': np.abs(scores)}

    with open(get_path(data_dir) / model_fname, 'wb') as f:
        pickle.dump(output, f)

    return output


def load_model_data(fname: str = 'model', data_dir: str = 'data'):
    ''' Load a machine learning model. '''
    from utils import get_path
    import pickle
    model_path = get_path(data_dir) / fname
    if model_path.is_file():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f'No model named {fname} found in {data_dir}.')

def predict_demand(date: str, api_key: str, model_fname: str = 'model',
    data_dir: str = 'data', return_std: bool = False) -> int:
    from datetime import datetime
    import pandas as pd
    from data import get_bristol_weather
    from utils import precip_type

    model = load_model_data(model_fname, data_dir = data_dir)['model']

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')

    weather_data = get_bristol_weather(date, api_key = api_key)
    weather_data['precip_type'] = precip_type(weather_data['precip_type'])

    date_data = {
        'month': date.month,
        'day_of_month': date.day,
        'day_of_week': date.isoweekday()
    }

    data_dict = {k: [v] for k, v in {**date_data, **weather_data}.items()}
    df = pd.DataFrame(data_dict)

    if return_std: 
        pred = np.around(model.predict(df, return_std = True))
        return tuple(pred.ravel().astype(int))
    else: 
        return np.around(model.predict(df)[0]).astype(int)

if __name__ == '__main__':
    from utils import get_path
    from data import get_data
    import numpy as np

    with open(get_path('darksky_key.txt'), 'r') as file_in:
        KEY = file_in.read().rstrip()

    #X, y = get_data()
    #train_gaussian_process(X, y)

    demand = predict_demand(
        date = '2020-01-12', 
        model_fname = 'gaussian_process', 
        api_key = KEY,
        return_std = True
    )
    print(demand)
