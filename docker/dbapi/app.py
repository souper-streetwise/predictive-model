from flask import Flask, request
import sqlite3
import json

app = Flask(__name__)

# Create tables
with sqlite3.connect('/soup.db') as db:
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
    cursor.close()

@app.route('/', methods = ['GET', 'POST'])
def data():

    # If someone performs a GET request then return all data from
    # the requested data, in a JSON format
    if request.method == 'GET':
        with sqlite3.connect('/soup.db') as db:
            cursor = db.cursor()
            table = request.args.get('table', 'counts')
            cursor.execute(f'SELECT * FROM {table}')
            data = list(cursor.fetchall())
            cursor.close()
        return json.dumps(data)

    # If someone performs a POST request then put all the data received
    # into the specified table, overwriting previous data on the same date
    elif request.method == 'POST':
        table = request.json['table']
        data = request.json['data']
        with sqlite3.connect('/soup.db') as db:
            cursor = db.cursor()
            for row in data:
                cursor.execute(\
                    f'''REPLACE INTO {table} 
                        VALUES(?{", ?" * (len(row) - 1)})''', 
                    row)
            db.commit()
            cursor.close()
        return '0'

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
