#=========================================================================
#
#  Script which is the main script of the ML container, will be run 
#  once every day.
#
#  The script pulls the database, trains a model, performs predictions
#  for the next 7 days, stores the predictions in the database along with
#  the weather data used to make those predictions, and finally saves the
#  model, which is linked to a model directory outside the container.
#
#=========================================================================

from .model import train_model
from .weather import get_bristol_weather, precip_type
from pathlib import Path
import datetime as dt
import pandas as pd
from collections import defaultdict
import requests

MODEL_NAME = 'soup_model'
DATA_DIR = '/models'

# Load weather data
weather_columns = ['date', 'precip_intensity_max', 'precip_intensity_avg',
                   'precip_type', 'wind_speed_max', 'wind_speed_avg',
                   'gust_max', 'gust_avg', 'temp_min', 'temp_max', 'temp_avg',
                   'temp_day', 'temp_night', 'humidity']
response = requests.get('http://dbapi:8080', params = {'table': 'weather'})
weather_df = pd.DataFrame(response.json(), columns = weather_columns)

# Load count data
counts_columns = ['date', 'fst_loc', 'snd_loc']
response = requests.get('http://dbapi:8080', params = {'table': 'counts'})
counts_df = pd.DataFrame(response.json(), columns = counts_columns)

# Merge weather and counts
df = weather_df.merge(counts_df, on = 'date')

# Create date data
datetimes = df.date.map(lambda txt: dt.datetime.strptime(txt, '%Y-%m-%d'))
df['year'] = [date.year for date in datetimes]
df['month'] = [date.month for date in datetimes]
df['day_of_week'] = [date.isoweekday() for date in datetimes]
df['day_of_month'] = [date.day for date in datetimes]

# Pull out relevant columns
X = df[['year', 'month', 'day_of_week', 'day_of_month', 
        'precip_intensity_avg', 'precip_type', 'wind_speed_avg', 
        'temp_avg', 'humidity']]
y = df.fst_loc + df.snd_loc
del df

# Train model
model_data = train_model(X, y, data_dir = DATA_DIR, model_name = MODEL_NAME)
model = model_data['model']

# Fetch DarkSky API key
with open(Path('ml') / 'darksky_key.txt', 'r') as f:
    api_key = f.read().rstrip()

# Get datetime objects
today = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')
week = [dt.datetime.now() + dt.timedelta(days = n) for n in range(0, 7)]
week = [dt.datetime.strftime(date, '%Y-%m-%d') for date in week]

prediction_dict = defaultdict(list)
weather_dict = defaultdict(list)
for date in week:

    # Set date data
    prediction_dict['date'].append(date)
    prediction_dict['date_predicted'].append(today)

    # Get weather data from the Dark Sky API
    weather_dict['date'].append(date)
    weather_data = get_bristol_weather(date, api_key = api_key)
    weather_data['precip_type'] = precip_type(weather_data['precip_type'])

    # Add weather data to weather dictionary
    for key, val in weather_data.items():
        weather_dict[key].append(val)

# Create dataframes
prediction_df = pd.DataFrame(prediction_dict)
weather_df = pd.DataFrame(weather_dict)

# Make temporary dataframe for predictions, and add date data to it
tmp = weather_df.copy()
datetimes = tmp.date.map(lambda txt: dt.datetime.strptime(txt, '%Y-%m-%d'))
tmp['year'] = [date.year for date in datetimes]
tmp['month'] = [date.month for date in datetimes]
tmp['day_of_week'] = [date.isoweekday() for date in datetimes]
tmp['day_of_month'] = [date.day for date in datetimes]

# Pull out relevant columns
X = tmp[['year', 'month', 'day_of_week', 'day_of_month', 
        'precip_intensity_avg', 'precip_type', 'wind_speed_avg', 
        'temp_avg', 'humidity']]
del tmp

# Add predictions
prediction_df['prediction'] = np.around(model.predict(X)).astype(int)
prediction_df['lower'] = np.around(model.predict(X, 2.5)).astype(int)
prediction_df['upper'] = np.around(model.predict(X, 97.5)).astype(int)

# Store prediction data into database
json = {'table': 'predictions', 'data': prediction_df.to_dict('split')['data']}
requests.post('http://dbapi:8080', json = json)

# Store weather data into database
json = {'table': 'weather', 'data': weather_df.to_dict('split')['data']}
requests.post('http://dbapi:8080', json = json)
