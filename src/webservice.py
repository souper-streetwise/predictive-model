from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return ''

@app.route('/predict')
def predict():
    from flask import request
    from datetime import datetime, timedelta
    from model import predict_demand
    import json

    api_key = request.args['api_key']
    model_fname = request.args.get('model_fname', 'random_forest')
    data_dir = request.args.get('data_dir', 'data')
    percentile = request.args.get('percentile', 90)
    date = request.args.get('date', datetime.now() + timedelta(days = 1))

    if isinstance(date, datetime): date = datetime.strftime(date, '%Y-%m-%d')

    prediction = predict_demand(
        date = date,
        api_key = api_key,
        model_fname = model_fname,
        data_dir = data_dir,
        percentile = percentile
    )

    # Checks if prediction is NaN
    if prediction != prediction: 
        error = 'We cannot predict more than nine days away!'
        result = {'date': date, 'error': error}

    else:
        if percentile is None:
            result = {'date': date, 'prediction': prediction}
        else:
            lower = prediction[0]
            median = prediction[1]
            upper = prediction[2]
            result = {
                'date': date, 
                'prediction': median, 
                'lower': lower,
                'upper': upper
            }
    
    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
