import pytest
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from ml.model import train_model, inference, save_model
from ml.data import process_data
# TODO: add necessary import


@pytest.fixture
def sample_data():
    """
    Create sample dataset for testing
    """
    data = {
        "age": [20, 30, 31],
        "race": ["White", "Black", "Other"],
        "sex": ["Male", "Female", "Tree"],
        "target": [0, 1, 0],
    }
    return pd.DataFrame(data)


# TODO: implement the first test. Change the function name and input as needed
def test_train_model(sample_data):
    """
    Test if the returned model is the correct type
    """
    categorical_features = ["race", "sex"]
    label = "target"

    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=categorical_features, label=label, training=True
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


# TODO: implement the second test. Change the function name and input as needed
@pytest.fixture
def sample_model():
    """
    Create a sample dataset and a trained model for testing
    """
    X, y = make_classification(n_samples=100, n_features=14, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X

def test_inference(sample_model):
    """
    Test the inference function using sample model
    """
    model, X = sample_model
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Preds are not a numpy array"
    assert preds.shape[0] == X.shape[0], "Preds does not match input"

# TODO: implement the third test. Change the function name and input as needed
def test_save_model(sample_model):
    """
    Test save_model function
    """
    model, _ = sample_model
    test_path = "test_model.pkl"
    save_model(model, test_path)
    assert os.path.exists(test_path), f"Model was not saved at {test_path}"
    os.remove(test_path)
