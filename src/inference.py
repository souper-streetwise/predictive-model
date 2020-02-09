from typing import Union, Sequence, Tuple

def predict_demand(date: str, api_key: str, data_dir: str = 'data', 
    alpha: float = .99, model_name: str = 'soup_model'
    ) -> Tuple[Sequence[float], Sequence[Tuple[float, float]]]:

    from datetime import datetime
    import pandas as pd
    import numpy as np
    from data import get_bristol_weather
    from utils import precip_type, load_model_data

    model = load_model_data(model_name, data_dir = data_dir)['model']

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
    if df.iloc[0, :].isna().any(): return None

    preds, intervals = model(df, return_cis = True, alpha = alpha)
    return preds, intervals

if __name__ == '__main__':
    from utils import get_path
    from datetime import datetime, timedelta
    from argparse import ArgumentParser

    tomorrow = datetime.today() + timedelta(days = 1)
    tomorrow = datetime.strftime(tomorrow, '%Y-%m-%d')

    parser = ArgumentParser()
    parser.add_argument('--date', default = tomorrow)
    args = vars(parser.parse_args())

    with open(get_path('darksky_key.txt'), 'r') as file_in:
        KEY = file_in.read().rstrip()

    demand = predict_demand(date = args['date'], api_key = KEY)
    print(demand)
