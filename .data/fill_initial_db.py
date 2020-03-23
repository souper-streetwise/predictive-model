import sqlite3
import pandas as pd
from ml.utils import precip_type

with sqlite3.connect('soup.db') as db:
    cursor = db.cursor()
    cursor.execute(\
    '''CREATE TABLE IF NOT EXISTS weather(
        date TEXT PRIMARY KEY,
        precip_intensity_max REAL,
        precip_intensity_avg REAL,
        precip_type INTEGER,
        wind_speed_max REAL,
        wind_speed_avg REAL,
        gust_max REAL,
        gust_avg REAL,
        temp_min REAL,
        temp_max REAL,
        temp_avg REAL,
        temp_day REAL,
        temp_night REAL,
        humidity REAL)''')
    cursor.execute(\
    '''CREATE TABLE IF NOT EXISTS counts(
        date TEXT PRIMARY KEY,
        fst_loc REAL,
        snd_loc REAL)''')
    cursor.execute(\
    '''CREATE TABLE IF NOT EXISTS predictions(
        date TEXT PRIMARY KEY,
        date_predicted TEXT,
        prediction REAL,
        lower REAL,
        upper REAL)''')

    weather_df = pd.read_csv('weather_data.tsv', sep = '\t')
    weather_df['precip_type'] = weather_df.precip_type.map(precip_type)
    for row in weather_df.iterrows():
        row = list(row[1])
        cursor.execute(\
            f'''REPLACE INTO weather 
                VALUES(?{", ?" * (len(row) - 1)})''', 
            row)

    counts_df = pd.read_csv('inital_data_march2020_update_clean.csv',
        usecols = ['date', 'fst', 'snd'])
    for row in counts_df.iterrows():
        row = list(row[1])
        cursor.execute(\
            f'''REPLACE INTO counts
                VALUES(?{", ?" * (len(row) - 1)})''', 
            row)

    db.commit()
