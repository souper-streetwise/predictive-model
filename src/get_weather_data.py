def get_bristol_weather(date, key):
    ''' Get weather data in Bristol at a particular date.
    
    INPUT
        date: str
            A date in the YYYY-MM-DD format
        key: str
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

    # Bristol's latitude and longitude coordinates
    lat, lng = (51.4545, 2.5879)
    
    # Perform a GET request from the Dark Sky API
    url = f'https://api.darksky.net/forecast/{key}/{lat},{lng},{date}T00:00:00'
    params = {
        'exclude': ['currently', 'minutely', 'alerts', 'flags'],
        'units': 'si'
        }
    response = requests.get(url, params = params)
    
    # Convert response to dictionary
    raw = json.loads(response.text)
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

def get_bristol_weathers(from_date, to_date, key):
    ''' Get all weather data in Bristol from a given date to a given date.
    
    INPUT
        from_date: str
            A date in the YYYY-MM-DD format
        to_date: str
            A date in the YYYY-MM-DD format
        key: str
            A Dark Sky API key, get one for free at https://darksky.net/dev

    OUTPUT
        A Pandas dataframe with all weather information in the given period
    '''
    from tqdm import trange
    import pandas as pd
    from calendar import monthrange

    # Pull out the years, months and days for convenience. We will change
    # the `to_month` to its correct values when we reach the last year
    from_year = int(from_date[:4])
    from_month = int(from_date[5:7])
    from_day = int(from_date[8:10])
    to_year = int(to_date[:4])
    to_month = 12

    data = {'day': [], 'month': [], 'year': []}
    for year in trange(from_year, to_year + 1, desc = 'Years'):

        # Start from January if it's not the first year, and only go to the
        # specified last month if it's the last year
        if year != from_year:
            from_month = 1
        if year == to_year:
            to_month = int(to_date[5:7])

        for month in trange(from_month, to_month + 1, desc = 'Months',
            leave = False):

            # Start from day 1 if it's not the first month, and only go to 
            # the specified last day if it's the last month of the last year.
            if month != from_month:
                from_day = 1
            if month == to_month and year == to_year:
                to_day = int(to_date[8:10])
            else:
                to_day = monthrange(year, month)[1]

            for day in trange(from_day, to_day + 1, desc = 'Days',
                leave = False):

                data['day'].append(day)
                data['month'].append(month)
                data['year'].append(year)

                # Get weather data
                # Note: Here {foo:0>2} prepends a 0 to foo if needed, to
                # ensure that it has two numerals
                weather = get_bristol_weather(
                    f'{year}-{month:0>2}-{day:0>2}', 
                    key = KEY
                    )

                # If this is the first entry then add the weather dictionary
                if data is None:
                    weather_data = {k: [v] for (k, v) in weather.items()}
                    data = {**data, **weather_data}

                # If not then append the dictionary with the new values
                else:
                    for key, val in weather.items():
                        data[key].append(val)

    # Convert the dictionary to a Pandas dataframe and return it
    return pd.DataFrame(data)


if __name__ == '__main__':
    with open('darksky_key.txt', 'r') as file_in:
        KEY = file_in.read().rstrip()

    weather = get_bristol_weathers('2018-02-03', '2019-11-03', key = KEY)
    weather.to_csv('data/weather_data.csv')
