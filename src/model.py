import pandas as pd
import joblib
from preprocessing.inference_features import InferenceFeatureBuilder
from utils.xgb_pipeline import XGBoostPipeline
from preprocessing.tree_preprocessor import create_tree_preprocessor
from sklearn.model_selection import train_test_split
from pathlib import Path
from optuna_integration import XGBoostPruningCallback
import xgboost as xgb
import numpy as np
import optuna
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


EMBEDDING_PCA_COMPONENTS=370

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "artifacts" / "model"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = MODEL_DIR / "final_model"

DATASET_PATH = BASE_DIR / "dataset" / "Airbnb_Data.csv"

DEVICE = 'cuda' if xgb.build_info()['USE_CUDA'] else 'cpu'
ALPHA = 0.03
N_TRIALS=250

def main():
    df = pd.read_csv(DATASET_PATH)

    builder = InferenceFeatureBuilder(
        use_amenities=True,
        use_embeddings=True,
        embedding_pca_components=EMBEDDING_PCA_COMPONENTS
    )

    builder.fit(df)
    df_copy = builder.transform(df)

    X = df_copy.drop(columns=['log_price'])
    y = df_copy['log_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=42,
        test_size=0.2
    )
    
    cat_features = X_train.select_dtypes(include=['string', 'object']).columns
    tree_preprocessor = create_tree_preprocessor(cat_features)

    X_train_processed = tree_preprocessor.fit_transform(X_train, y_train)
    X_test_processed = tree_preprocessor.transform(X_test)

    X_train_processed = X_train_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)

    dtrain = xgb.DMatrix(X_train_processed, label=y_train)

    xgb_pipeline = train_model(
        dtrain=dtrain,
        preprocessor=tree_preprocessor,
        X_train=X_train,
        y_train=y_train
    )

    pred_train_log = xgb_pipeline.predict(X_train)
    pred_test_log = xgb_pipeline.predict(X_test)

    evaluate(
        pred_train_log=pred_train_log,
        pred_test_log=pred_test_log,
        y_train=y_train,
        y_test=y_test
    )

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


def train_model(dtrain, preprocessor, X_train, y_train):

    model_exists = (
        MODEL_PATH.with_suffix(".json").exists() and
        MODEL_PATH.with_suffix(".joblib").exists() and
        MODEL_PATH.with_suffix(".params").exists()
    )

    if model_exists:
        print(f"Loading existing model...")
        xgb_pipeline = XGBoostPipeline.load(MODEL_PATH)
    else:
        print(f"Training model...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=10)
        )

        study.optimize(
            lambda trial: objective(trial, dtrain, alpha=ALPHA),
            n_trials=N_TRIALS,
            gc_after_trial=True,
            callbacks=[optuna_callback]
        )

        best_num_boost_rounds = study.best_trial.user_attrs['best_num_boost_round']

        best_params = {
            **study.best_params,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "n_jobs": -1,
            "seed": 42,
            "verbosity": 0,
            "device": DEVICE
        }

        xgb_pipeline = XGBoostPipeline(preprocessor=preprocessor)
        xgb_pipeline.fit(
            X_train, y_train,
            params=best_params,
            num_boost_round=best_num_boost_rounds
        )

        xgb_pipeline.save(MODEL_PATH)

    return xgb_pipeline
    
def objective(trial, dtrain, alpha=0.0):

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": DEVICE,
        
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 20, log=True),

        "verbosity": 0,
        "n_jobs": -1,
        "seed": 42
    }

    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        early_stopping_rounds=50,
        num_boost_round=1000,
        nfold=3,
        metrics="rmse",
        seed=42,
        verbose_eval=False,
        shuffle=True,
        callbacks=[XGBoostPruningCallback(trial, "test-rmse")]
    )

    trial.set_user_attr(
        "best_num_boost_round",
        len(cv_results)
    )

    test_rmse = cv_results['test-rmse-mean'].iloc[-1]
    train_rmse = cv_results['train-rmse-mean'].iloc[-1]

    gap = (test_rmse - train_rmse) / test_rmse
    
    trial.set_user_attr("gap", gap)
    trial.set_user_attr("train_rmse", train_rmse)
    trial.set_user_attr("test_rmse", test_rmse)

    score = test_rmse + alpha * gap

    return score

def optuna_callback(study, trial):
    print(
        f"Trial {trial.number + 1}/{N_TRIALS} "
        f"| Score: {trial.value:.4f} "
        f"| Best: {study.best_value:.4f}"
    )

if __name__ == "__main__":
    main()

