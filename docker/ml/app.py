from flask import Flask, request
from datetime import datetime, timedelta
import json
from pathlib import Path
from .inference import predict_demand

app = Flask(__name__)

@app.route('/predict')
def predict():

    # Fetch API key, or default to the one stored in darksky_key.txt
    api_key = request.args.get('api_key')
    if api_key is None:
        with open(Path('api') / 'darksky_key.txt', 'r') as f:
            api_key = f.read().rstrip()

    # Fetch date, data_dir and the uncertainty level alpha
    tomorrow = datetime.now() + timedelta(days = 1)
    date = request.args.get('date', datetime.strftime(tomorrow, '%Y-%m-%d'))
    data_dir = request.args.get('data_dir', Path('api') / 'models')
    alpha = request.args.get('alpha', .95)

    prediction = predict_demand(
        date = date,
        api_key = api_key,
        data_dir = data_dir,
        alpha = alpha,
        model_name = 'soup_model'
    )

    if prediction is None:
        error = 'We cannot predict more than nine days away!'
        result = {'date': date, 'error': error}

    else:
        if alpha is None:
            result = {'prediction': prediction}
        else:
            lower = prediction[0]
            upper = prediction[2]
            prediction = prediction[1]
            result = {
                'date': date,
                'prediction': int(prediction),
                'lower': int(lower),
                'upper': int(upper)
            }

    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
