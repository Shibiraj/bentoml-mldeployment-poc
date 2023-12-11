"""
This module contains the classifier model
"""

from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split


def get_classifier_model():
    """Returns the classifier Model."""
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    return classifier
