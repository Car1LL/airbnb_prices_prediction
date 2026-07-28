import xgboost as xgb
import joblib
from pathlib import Path

class XGBoostPipeline:
    def __init__(self, preprocessor=None, booster=None, params=None):
        self.preprocessor = preprocessor
        self.booster = booster
        self.params = params

    def fit(self, X_train, y_train, X_val, y_val, params):
        self.preprocessor.fit(X_train, y_train)
        self.params = params.copy()

        X_train = self.preprocessor.transform(X_train)
        X_val = self.preprocessor.transform(X_val)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dval, "validation")],
            early_stopping_rounds=30,
            verbose_eval=False  
        )

        return self

    def predict(self, X):
        X = self.preprocessor.transform(X)
        dmatrix = xgb.DMatrix(X)

        return self.booster.predict(dmatrix)

    def save(self, path):
        path = Path(path)

        self.booster.save_model(
            path.with_suffix(".json")
        )

        joblib.dump(
            self.preprocessor,
            path.with_suffix(".joblib")
        )

        joblib.dump(
            self.params,
            path.with_suffix(".params")
        )

    @classmethod
    def load(cls, path):
        path = Path(path)

        booster = xgb.Booster()
        booster.load_model(
            path.with_suffix(".json")
        )

        preprocessor = joblib.load(
            path.with_suffix(".joblib")
        )

        params = joblib.load(
            path.with_suffix(".params")
        )

        return cls(
            preprocessor=preprocessor,
            booster=booster,
            params=params
        )
