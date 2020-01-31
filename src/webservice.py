from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return ''

@app.route('/soup', methods = ['POST', 'GET'])
def soup():
    from datetime import datetime, timedelta
    from inference import predict_demand
    import json
    import sys

    sys.path.append('soup')

    data_dict = request.form if request.method == 'POST' else request.args
    if not data_dict:
        return render_template('soup.html')

    if data_dict.get('api_key') is None:
        with open(Path('soup') / 'darksky_key.txt', 'r') as f:
            api_key = f.read().rstrip()
    else:
        api_key = data_dict['api_key']

    data_dir = data_dict.get('data_dir', '.data')
    percentile = data_dict.get('percentile', 90)
    if data_dict.get('date') is None:
        data_dict['date'] = datetime.now() + timedelta(days = 1)

    if isinstance(data_dict['date'], datetime):
        data_dict['date'] = datetime.strftime(data_dict['date'], '%Y-%m-%d')

    prediction = predict_demand(
        date = data_dict['date'],
        api_key = api_key,
        data_dir = data_dir,
        percentile = percentile
    )

    if prediction is None:
        error = 'We cannot predict more than nine days away!'
        result = {'date': data_dict['date'], 'error': error}

    else:
        if percentile is None:
            result = {'prediction': prediction}
        else:
            lower = prediction[0]
            upper = prediction[2]
            prediction = prediction[1]
            result = {
                'prediction': prediction,
                'lower': lower,
                'upper': upper
            }

    if request.method == 'POST':
        return render_template('soup.html', **result, **data_dict)
    else:
        return json.dumps(result)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
