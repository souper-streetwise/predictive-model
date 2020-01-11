def get_data(fname: str = 'dataset', data_dir: str = 'data', 
    normalise: bool = True, include_date: bool = False):
    import pandas as pd
    from utils import get_path, precip_type, month, day_of_week

    df = pd.read_csv(get_path(data_dir) / f'{fname}.tsv', sep = '\t', 
        header = 0).drop(columns = ['locA', 'locB'])

    if not include_date: df.drop(columns = ['date'], inplace = True)

    df['precip_type'] = df['precip_type'].map(precip_type)
    df['month'] = df['month'].map(month)
    df['day_of_week'] = df['day_of_week'].map(day_of_week)

    X = df[[col for col in df.columns if col != 'total']].copy()
    y = df['total']

    return X, y

def build_data(api_key: str, raw_fname: str = 'initial_data_no_duplicates.csv',
    out_fname: str = 'dataset', data_dir: str = 'data',
    weather_fname: str = 'weather_data.tsv'):
    import pandas as pd
    from datetime import datetime
    from utils import get_path

    col_names = ['date', 'locA', 'locB']
    raw_df = pd.read_csv(
        get_path(data_dir) / raw_fname, 
        names = col_names, 
        header = 0
    )

    raw_df['date'] = [datetime.strptime(date, '%m/%d/%Y')
                      for date in raw_df['date']] 

    raw_df['total'] = raw_df['locA'] + raw_df['locB']
    date_df = extract_date_data(raw_df['date'])

    update_past_weather_data(api_key = api_key)
    weather_df = extract_past_weather_data(raw_df['date'], data_dir = data_dir)
    weather_df['date'] = [datetime.strptime(date, '%Y-%m-%d')
                          for date in weather_df['date']]

    out_path = get_path(data_dir) / f'{out_fname}.tsv'
    df = pd.concat([raw_df, date_df], axis = 1)
    df = df.merge(weather_df, on = 'date')
    df.to_csv(out_path, sep = '\t', index = False)
    return df

def extract_date_data(dates: list):
    from datetime import datetime
    import pandas as pd
    from utils import day_of_week, month
    date_data = {
        'month': [month(date.month) for date in dates],
        'day_of_month': [date.day for date in dates],
        'day_of_week': [day_of_week(date.isoweekday()) for date in dates],
    }
    return pd.DataFrame(date_data)

def extract_past_weather_data(dates: list, api_key: str, 
    fname: str = 'weather_data.tsv', data_dir: str = 'data'):
    import pandas as pd
    from utils import get_path

    path = get_path(data_dir) / fname
    weather_df = pd.read_csv(path, sep = '\t')
    dates = [date.strftime('%Y-%m-%d') for date in dates]
    return weather_df[weather_df['date'].isin(dates)]

def update_past_weather_data(api_key: str, fname: str = 'weather_data.tsv', 
    data_dir: str = 'data'):
    import pandas as pd
    from datetime import datetime, timedelta
    from tqdm.auto import tqdm
    from utils import get_path, get_dates

    path = get_path(data_dir) / fname
    weather_df = pd.read_csv(path, sep = '\t')

    last_row = weather_df.tail(1).reset_index()
    last_date = datetime.strptime(last_row['date'][0], '%Y-%m-%d')

    if last_date.date() != datetime.today().date():
        dates = get_dates(last_date + timedelta(days = 1), datetime.today())
        rows = []
        for date in tqdm(dates, desc = 'Fetching weather data'):
            row = {'date': date.strftime('%Y-%m-%d')}
            row.update(get_bristol_weather(date, api_key))
            rows.append(row)
        weather_df = pd.concat([weather_df, pd.DataFrame(rows)], axis = 0)
        weather_df['precip_type'].fillna('no_precip', inplace = True)
        weather_df.to_csv(path, sep = '\t', index = False)

def get_bristol_weather(date: str, api_key: str):
    ''' Get weather data in Bristol at a particular date.
    
    INPUT
        date: datetime.datetime or str
            A given date. If a string is provided then it must be of the
            form 'YYYY-MM-DD'
        api_key: str
            A Dark Sky API key, get one for free at https://darksky.net/dev
            
    OUTPUT
        A dictionary containing:
            precip_intensity_max: The maximum precipitation intensity,
                                  measured in liquid water per hour
            precip_intensity_avg: The average precipitation intensity,
                                  measured in liquid water per hour
            precip_type: Type of precipitation, can be rain, snow or sleet
            wind_speed_max: The maximum wind speed, measured in m/s
            wind_speed_avg: The average wind speed, measured in m/s
            gust_max: The maximum gust speed, measured in m/s
            gust_avg: The average gust speed, measured in m/s
            temp_min: The minimum feel-like temperature, in celsius
            temp_max: The maximum feel-like temperature, in celsius
            temp_avg: The average feel-like temperature, in celsius
            temp_day: The feel-like temperature at midday, in celsius
            temp_night: The feel-like temperature at midnight, in celsius
            humidity: The relative humidity between 0 and 1, inclusive
    '''
    import requests
    import json
    import numpy as np
    from datetime import datetime

    # Convert datetime object to date string of the form YYYY-MM-DD
    if isinstance(date, datetime):
        date = date.strftime('%Y-%m-%d')

    # Bristol's latitude and longitude coordinates
    lat, lng = (51.4545, 2.5879)

    # Perform a GET request from the Dark Sky API
    url = f'https://api.darksky.net/forecast/'\
          f'{api_key}/{lat},{lng},{date}T00:00:00'
    params = {
        'exclude': ['currently', 'minutely', 'alerts', 'flags'],
        'units': 'si'
        }
    response = requests.get(url, params = params)
    
    # Convert response to dictionary
    raw = json.loads(response.text)

    # Check if an error occured
    while 'error' in raw.keys():
        raise Exception(raw['error'])

    # Pull out hourly and daily data
    hourly = raw['hourly']['data']
    daily = raw['daily']['data'][0]

    # Calculate averages
    precip_intensity_avg = np.around(np.mean([hour.get('precipIntensity') 
        for hour in hourly if hour.get('precipIntensity') is not None]), 4)
    wind_speed_avg = np.around(np.mean([hour.get('windSpeed')
        for hour in hourly if hour.get('windSpeed') is not None]), 2)
    gust_avg = np.around(np.mean([hour.get('windGust')
        for hour in hourly if hour.get('windGust') is not None]), 2)
    temp_avg = np.around(np.mean([hour.get('apparentTemperature')
        for hour in hourly if hour.get('apparentTemperature') is not None]), 2)

    data = {
        'precip_intensity_max': daily.get('precipIntensityMax'),
        'precip_intensity_avg': precip_intensity_avg,
        'precip_type': daily.get('precipType'),
        'wind_speed_max': daily.get('windSpeed'),
        'wind_speed_avg': wind_speed_avg,
        'gust_max': daily.get('windGust'),
        'gust_avg': gust_avg,
        'temp_min': daily.get('apparentTemperatureMin'),
        'temp_max': daily.get('apparentTemperatureMax'),
        'temp_avg': temp_avg,
        'temp_day': daily.get('apparentTemperatureHigh'),
        'temp_night': daily.get('apparentTemperatureLow'),
        'humidity': daily.get('humidity')
        }

    return data


if __name__ == '__main__':
    from utils import get_path
    with open(get_path('darksky_key.txt'), 'r') as file_in:
        KEY = file_in.read().rstrip()

    #build_data(api_key = KEY)
    print(get_data())
