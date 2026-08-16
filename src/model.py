import pandas as pd
from preprocessing.inference_features import InferenceFeatureBuilder
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
import joblib


EMBEDDING_PCA_COMPONENTS=370
DEVICE = "GPU" if get_gpu_device_count() > 0 else "CPU"

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "artifacts" / "model"
BUILDER_PATH = MODEL_DIR / "builder.pkl"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = MODEL_DIR / "final_model.cbm"
DATASET_PATH = BASE_DIR / "dataset" / "Airbnb_Data.csv"

def main():
    get_evaluate_model()

def get_evaluate_model():
    df = pd.read_csv(DATASET_PATH)
    df_copy = df.copy()

    X = df_copy.drop(columns=['log_price'])
    y = df_copy['log_price']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        random_state=42,
        test_size=0.2
    )

    builder = InferenceFeatureBuilder(
        use_amenities=True,
        use_embeddings=True,
        embedding_pca_components=EMBEDDING_PCA_COMPONENTS
    )

    builder.fit(X_train_raw)

    # Save builder 
    joblib.dump(builder, BUILDER_PATH)
    print(f"Feature Builder was successfully saved at: {BUILDER_PATH}")

    X_train = builder.transform(X_train_raw)
    X_test = builder.transform(X_test_raw)

    cat_features = X_train.select_dtypes(include=['string', 'object']).columns.tolist()

    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    test_pool = Pool(X_test, label=y_test, cat_features=cat_features)

    catboost_pipeline = get_model(train_pool)

    pred_train_log = catboost_pipeline.predict(train_pool)
    pred_test_log = catboost_pipeline.predict(test_pool)

    evaluate(
        pred_train_log=pred_train_log,
        pred_test_log=pred_test_log,
        y_train=y_train,
        y_test=y_test
    )

    print(f"\n\nModel is located at: {MODEL_PATH}")

def ensure_model():
    if MODEL_PATH.exists() and BUILDER_PATH.exists():
        print(f"Model and feature builder already exist.")
        return
    print(f"Model artifacts not found, Starting training")
    get_evaluate_model()
    
def train_model(train_pool):
    print(f"Training the model on device: {DEVICE}")

    catboost_pipeline = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        task_type=DEVICE
    )

    catboost_pipeline.fit(train_pool)
    catboost_pipeline.save_model(MODEL_PATH)
    print(f"Model was successfully trained and saved at: {MODEL_PATH}")

    return catboost_pipeline

def get_model(train_pool):
    if MODEL_PATH.exists():
        print(f"Loading existing model from: {MODEL_PATH}")

        model = CatBoostRegressor()
        model.load_model(MODEL_PATH)

        return model
    
    return train_model(train_pool)


def evaluate(pred_train_log, pred_test_log, y_train, y_test):
    pred_train = np.exp(pred_train_log)
    pred_test = np.exp(pred_test_log)

    test_MAE = mean_absolute_error(np.exp(y_test), pred_test)
    train_MAE = mean_absolute_error(np.exp(y_train), pred_train)

    test_RMSE = root_mean_squared_error(np.exp(y_test), pred_test)
    train_RMSE = root_mean_squared_error(np.exp(y_train), pred_train)

    test_r2 = r2_score(y_test, pred_test_log)
    train_r2 = r2_score(y_train, pred_train_log)

    print("\n\n")
    print(" Evaluation ".center(30, "="))
    print(f"Test MAE: {test_MAE:.2f}$ | Train MAE: {train_MAE:.2f}$")
    print(f"Test RMSE: {test_RMSE:.2f}$ | Train RMSE: {train_RMSE:.2f}$")
    print(f"Test R2 Score: {test_r2:.2f} | Train R2 Score: {train_r2:.2f}")

if __name__ == "__main__":
    main()