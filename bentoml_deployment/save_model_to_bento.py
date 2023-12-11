"""
This module saves a classifier model to BentoML.
"""

import bentoml
from model import get_classifier_model


def load_model_and_save_it_to_bento() -> None:
    """Loads a classifier model and saves it to BentoML."""
    classifier = get_classifier_model()
    saved_model = bentoml.sklearn.save_model("iris_classifier", classifier)
    print(f"Bento model tag = {saved_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento()
