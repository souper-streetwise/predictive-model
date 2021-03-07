import numpy as np
import datetime as dt
import requests
import json
from typing import Union, Dict

def precip_type(inputs: Union[int, str]) -> Union[int, str]:
    ''' Converts precipitation type between being an integer
        and a string.

    INPUT
        inputs: Union[int, str]
            Either a string being 'no_precip', 'rain', 'snow'
            or 'sleet', or an integer in the interval [0, 3].

    OUTPUT
        If input was an integer then output the corresponding
        string, and if input was a string the output the
        corresponding integer.
    '''
    precips = ['no_precip', 'rain', 'snow', 'sleet']
    if isinstance(inputs, int):
        return precips[inputs]
    else:
        if inputs is None: inputs = 'no_precip'
        return precips.index(inputs)

def get_bristol_weather(date: Union[dt.datetime, str], api_key: str)\
    -> Dict[str, Union[str, float, None]]:
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

    # Convert datetime object to date string of the form YYYY-MM-DD
    if isinstance(date, dt.datetime):
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
