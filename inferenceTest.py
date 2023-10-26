import requests
import json

url = 'https://onnxmodel-rhods-project.apps.io.mos-paas.de.eviden.com/v2/models/onnxmodel/infer'

input_data = {
    "date": '2013-01-06',
    "family": 'AUTOMOTIVE',
    "onpromotion": 0,
    "day_of_week_index": 1,
    "month_index": 1,
    "dcoilwtico": 91.11,
    "transactions": 90464,
    "type": 'Normal',
    "lag_1": 342.0,
    "lag_2": 169.0,
    "lag_3": 161.0,
}

response = requests.post(url, data=json.dumps({"instances": [input_data]}))

if response.status_code == 200:
    output_data = response.json()
    print(output_data)
else:
    print(f"Error: {response.status_code}, {response.text}")