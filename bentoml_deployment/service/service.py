"""
This module defines a BentoML service that uses the model to classify
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

BENTO_MODEL_TAG = "iris_classifier:latest"

classifier_runner = bentoml.sklearn.get(BENTO_MODEL_TAG).to_runner()

iris_service = bentoml.Service("iris_classifier", runners=[classifier_runner])


@iris_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    return classifier_runner.predict.run(input_data)
