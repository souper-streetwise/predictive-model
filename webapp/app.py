from flask import Flask, render_template
import requests

app = Flask(__name__)

@app.route('/')
def index():
    params = {'table': 'predictions'}
    response = requests.get('http://dbapi:8080', params = params)
    return render_template('index.html', data = response.json())
