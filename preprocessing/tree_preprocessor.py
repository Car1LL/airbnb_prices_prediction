from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def create_tree_preprocessor(cat_features):
    
    cat_transformer = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer,  cat_features)
        ],
        remainder="passthrough"
    )

    return preprocessor