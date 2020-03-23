import requests
import json

with open('/models/model_data.json', 'r') as f:
    data = json.load(f)

response = requests.post('http://dbapi:8080', data = json.dumps(data))
