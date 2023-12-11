"""
Service request for testing the API
"""
from typing import Tuple
import json

import numpy as np
import requests

SERVICE_URL = "http://localhost:3000/classify"


def sample_random_iris_data_point() -> Tuple[np.array, np.array]:
    return np.array([4.8, 3., 1.4, 0.1]), np.array([0])


def make_request_to_bento_service(
        service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps([input_array.tolist()])
    print(serialized_input_data)
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    print(response)
    return response.text


def main():
    input_data, expected_output = sample_random_iris_data_point()
    prediction = make_request_to_bento_service(SERVICE_URL, input_data)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()
