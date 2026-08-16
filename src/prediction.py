from src.model import get_evaluate_model, MODEL_PATH, BUILDER_PATH, DATASET_PATH, ensure_model
from catboost import Pool
import pandas as pd
import numpy as np
import joblib
from preprocessing.inference_features import InferenceFeatureBuilder
from catboost import CatBoostRegressor

def initialize():
    ensure_model()

    global builder
    global model

    builder = joblib.load(BUILDER_PATH)

    if builder.use_embeddings:
        builder.description_preprocessor.use_cache = False

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

def predict(data):
    df = pd.DataFrame([data])
    X = builder.transform(df)

    cat_features = X.select_dtypes(include=['string', 'object']).columns.tolist()
    pool = Pool(X, cat_features=cat_features)

    prediction_log = model.predict(pool)
    prediction = np.exp(prediction_log[0])

    return prediction