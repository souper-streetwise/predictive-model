def train_model(X, y, cv: int = 10, n_iter: int = 1,
    return_feature_importances: bool = False, data_dir: str = 'data'):
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

    output = {
        'model': model, 
        'scores': np.abs(cross_val_score(model, X, y, cv = cv)), 
        'params': rnd_search.best_params_
    }

    if return_feature_importances:
        feat_dict = dict(list(zip(model.feature_importances_, X.columns)))
        feat_sorted = sorted(feat_dict, reverse = True)
        feat_imps = [(feat_dict[imp], imp) for imp in feat_sorted]
        output['feat_imps'] = feat_imps

    with open(get_path(data_dir) / 'model', 'wb') as f:
        pickle.dump(output, f)

    return output

def predict_demand(date: str, api_key: str, model = None,
    data_dir: str = 'data') -> int:
    from datetime import datetime
    import pandas as pd
    import pickle
    from data import get_bristol_weather
    from utils import precip_type, get_path

    if model is None:
        model_path = get_path(data_dir) / 'model'
        if model_path.is_file():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)['model']
        else:
            raise FileNotFoundError('No model found in data folder.')

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
    return int(model.predict(df)[0])

if __name__ == '__main__':
    from data import get_data
    from utils import get_path

    with open(get_path('darksky_key.txt'), 'r') as file_in:
        KEY = file_in.read().rstrip()

    X, y = get_data()
    T = train_model(X, y, return_feature_importances = False)
    print(predict_demand('2020-01-12', api_key = KEY))
