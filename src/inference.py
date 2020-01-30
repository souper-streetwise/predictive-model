def predict_demand(date: str, api_key: str, data_dir: str = 'data', 
    percentile: int = 90) -> int:
    from datetime import datetime
    import pandas as pd
    from data import get_bristol_weather
    from utils import precip_type
    import numpy as np

    model = load_model_data(data_dir = data_dir)['model']

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
    if df.iloc[0, :].isna().any(): return np.nan

    percentiles = [50 - percentile / 2, 50, 50 + percentile / 2]
    preds = np.around(model.predict(df, percentiles = percentiles), 2)
    return preds.ravel() if percentiles is not None else preds[0]

if __name__ == '__main__':
    from utils import get_path

    with open(get_path('darksky_key.txt'), 'r') as file_in:
        KEY = file_in.read().rstrip()

    demand = predict_demand(date = '2020-01-23', api_key = KEY)
    print(demand)
