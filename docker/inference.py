from typing import Union, Sequence, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from .data import get_bristol_weather
from .utils import precip_type, load_model_data

def predict_demand(date: str, api_key: str, data_dir: str = 'data', 
    alpha: float = .95, model_name: str = 'soup_model'
    ) -> Tuple[Sequence[float], Sequence[Tuple[float, float]]]:

    model = load_model_data(model_name, data_dir = data_dir)['model']

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')

    weather_data = get_bristol_weather(date, api_key = api_key)

    # Choose only the independent features
    independent_feats = ['date', 'precip_intensity_avg', 'precip_type',
        'wind_speed_avg', 'temp_avg', 'humidity']
    redundant_feats = [feat for feat in weather_data.keys() 
                       if not feat in independent_feats]
    for feat in redundant_feats: weather_data.pop(feat)

    weather_data['precip_type'] = precip_type(weather_data['precip_type'])

    date_data = {
        'year': date.year,
        'month': date.month,
        'day_of_week': date.isoweekday(),
        'day_of_month': date.day,
    }

    data_dict = {k: [v] for k, v in {**date_data, **weather_data}.items()}
    df = pd.DataFrame(data_dict)
    if df.iloc[0, :].isna().any(): return None

    preds = model.predict(df)
    lower = model.predict(df, 100 * (1 - alpha) / 2)
    upper = model.predict(df, 100 * (1 + alpha) / 2)

    return lower, preds, upper

if __name__ == '__main__':
    pass
