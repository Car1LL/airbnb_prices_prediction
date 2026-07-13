import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def create_linear_preprocessor(cat_features, num_features):
    
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_features),
            ("num", num_transformer, num_features)
        ]
    )

    return preprocessor