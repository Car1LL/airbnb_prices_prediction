from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_tree_preprocessor(cat_features):
    
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer,  cat_features)
        ],
        remainder="passthrough"
    )

    return preprocessor