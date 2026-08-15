from utils.xgb_pipeline import XGBoostPipeline
import joblib
from pathlib import Path

class InferenceXGBoostPipeline(XGBoostPipeline):
    def __init__(self, preprocessor=None, booster=None, params=None, feature_builder=None):
        super().__init__(preprocessor, booster, params)
        self.feature_builder = feature_builder

    def predict(self, X):
        X = self.feature_builder.transform(X)

        return super().predict(X)

    def save(self, path):
        path = Path(path)

        super().save(path)

        joblib.dump(
            self.feature_builder,
            path.parent / f"{path.name}_feature_builder.joblib"
        )

    @classmethod
    def load(cls, path):
        path = Path(path)

        pipeline = super().load(path)

        feature_builder = joblib.load(
            path.parent / f"{path.name}_feature_builder.joblib"
        )

        return cls(
            feature_builder=feature_builder,
            preprocessor=pipeline.preprocessor,
            booster=pipeline.booster,
            params=pipeline.params
        )