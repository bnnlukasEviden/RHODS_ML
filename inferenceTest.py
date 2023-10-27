import requests
import json
import numpy as np

url = 'https://tf-model-lstm-rhods-project.apps.io.mos-paas.de.eviden.com/v2/models/tf-model-lstm/infer'

input_data = np.array([255., 161., 169., 342., 360., 189., 229., 164., 164., 162.]).tolist()
 
response = requests.post(url, data=json.dumps({"instances": [input_data]}))
 
if response.status_code == 200:
    output_data = response.json()
    print(output_data)
else:
    print(f"Error: {response.status_code}, {response.text}")