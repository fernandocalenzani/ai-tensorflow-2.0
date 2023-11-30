import json

import matplotlib.pyplot as plt
import numpy as np
import requests

x_test = 3000
class_names = ['airplane', 'automobile', 'bird',
               'cat', 'dog', 'frog', 'horse', 'ship', 'truck']


random_image = np.random.randint(0, len(x_test))

data = json.dumps({
    "signature_name": "serving_default",
    "instance": [x_test[random_image].tolist()]
})

headers = {
    "Content-Type": "application/json"
}

json_response = requests.post(
    url="http://localhost:8501/v1/models/cifar10:predict", data=data, headers=headers
)

predictions = json.load(json_response.text)["predictions"]
class_names[np.argmax(predictions)]

plt.imshow(x_test[random_image])


specific_json_response = requests.post(
    url="http://localhost:8501/v1/models/cifar10/versions/1.0.0:predict", data=data, headers=headers
)

predictions = json.load(specific_json_response.text)["predictions"]
class_names[np.argmax(predictions)]

plt.imshow(x_test[random_image])
