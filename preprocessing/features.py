import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch


ROOT = Path.cwd().parent.resolve()
DATASET_PATH = ROOT / "dataset" / "Airbnb_Data.csv"
EMBEDDINGS_PATH = ROOT / "dataset" / "description-embeddings.parquet"
EARTH_RADIUS = 6371
TOP_NEIGHBOURHOODS = 209
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AMENITIES_REPLACEMENTS = {
    "Firm matress": "Firm mattress",
    "Elevator in building": "Elevator",
    "Smartlock": "Smart lock",
    "Doorman Entry": "Doorman",
    "Wide clearance to shower & toilet": "Wide clearance to shower and toilet",
    "smooth pathway to front door": "Flat smooth pathway to front door"
}

AMENITIES_REMOVE = {
    "Flat",
}

AMENITIES_GROUPS = {
    "Bathroom supplies": {
        "Body soap",
        "Hand soap",
        "Bath towel",
        "Hand or paper towel",
        "Toilet paper"
    }
}

DESCRIPTION_GARBAGE_OBSERVATION = {
    '.',
    '...',
    '/',
    'Dasfdfads',
    'TEST!!',
    'Test',
    'Testing',
    'Thanks!',
    'a',
    'asdfsafda',
    'deleted',
    'f',
    'jkl',
    'n',
    'none',
    's',
    'sdsads',
    'x'
}

BOOLEAN_FIX = {
    "instant_bookable",
    "host_identity_verified"
}

def fill_host_response_rate(df):
    df['host_response_rate'] = pd.to_numeric(
        df['host_response_rate'].str.replace("%", "", regex=False),
        errors="coerce"
    )

    df['host_response_rate_missing'] = df['host_response_rate'] \
        .isna() \
        .astype("int8")

    median = df['host_response_rate'].median()

    df['host_response_rate'] = df['host_response_rate'].fillna(median)

    return df

def fill_categorical_na_values(df):
    categorical_features = df.select_dtypes(
        include=["object", "string"]
    ).columns

    excluded_columns = {
        "host_response_rate",
        "host_since",
        "last_review",
        "first_review"
    }

    for column in categorical_features:
        if column in excluded_columns:
            continue

        df[column] = df[column].fillna("Unknown")

    return df

def fill_numerical_na_values(df):
    numerical_features = df.select_dtypes(
        include=['float', 'int']
    ).columns

    excluded_columns = {
        "host_response_rate",
    }

    for column in numerical_features:
        if column in excluded_columns:
            continue
        df[column] = df[column].fillna(df[column].median())

    return df


def create_datetime_features(df):
    date_features = {
        "last_review",
        "first_review",
        "host_since"
    }

    for col in date_features:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    host_reference_date = df['host_since'].max()
    df['host_age_days'] = (host_reference_date - df['host_since']).dt.days

    last_review_reference_date = df['last_review'].max()
    df['days_since_last_review'] = (last_review_reference_date - df['last_review']).dt.days

    df['review_history_days'] = (
        df['last_review'] - df['first_review']
    ).dt.days

    df['first_review_year'] = df['first_review'].dt.year
    df['host_since_year'] = df['host_since'].dt.year

    df['has_reviews'] = (
        df['first_review'].notna()
    ).astype("int8")

    df['review_before_host'] = (
        df['first_review'] < df['host_since']
    ).fillna(False).astype("int8")

    df.drop(columns=[
        "first_review",
        "last_review",
        "host_since"
    ], inplace=True)

    return df

def fill_datetime_na_values(df):
    datetime_features = [
        "host_age_days",
        "host_since_year",
        "review_history_days",
        "days_since_last_review",
        "first_review_year"
    ]

    for col in datetime_features:
        df[col] = df[col].fillna(df[col].median())

    return df

def fix_beds(df):
    invalid_beds = df['beds'] == 0
    single_guest = df['accommodates'] == 1
    multiple_guests = df['accommodates'] > 1

    df.loc[invalid_beds & single_guest, 'beds'] = 1

    df.loc[invalid_beds & multiple_guests, 'beds'] = (
        df.loc[invalid_beds & multiple_guests, 'accommodates'] - 1
    )

    return df

def fix_boolean_features(df):
    mappings = {
        "t": True,
        "f": False
    }

    for col in BOOLEAN_FIX:
        df[col] = df[col].map(mappings)
        df[col] = df[col].fillna(-1).astype('int8')

    boolean_features = df.select_dtypes(include=['bool']).columns
    df[boolean_features] = df[boolean_features].astype('int8')

    return df

def create_distance_to_listing_center(df):
    
    listing_centers = df.groupby('city')[['latitude', 'longitude']] \
        .median() \
        .rename(columns={
            "latitude": "listing_center_lat",
            "longitude": "listing_center_lon"
        })
    
    df = df.merge(listing_centers, on='city', how='left')
    
    lat1 = np.radians(df['latitude'])
    lon1 = np.radians(df['longitude'])

    lat2 = np.radians(df['listing_center_lat'])
    lon2 = np.radians(df['listing_center_lon'])

    def haversine(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.asin(np.sqrt(a))

        return EARTH_RADIUS * c
    
    df['distance_to_listing_center'] = haversine(lat1, lon1, lat2, lon2)

    df.drop(columns=[
        "listing_center_lat",
        "listing_center_lon"
    ], inplace=True)

    return df

def group_rare_property_types(df):
    property_counts = df['property_type'].value_counts()

    rare_categories = property_counts[property_counts < 10].index

    df['property_type'] = df['property_type'].replace(
        rare_categories,
        "Other"
    )

    return df

def group_rare_bed_types(df):
    real_bed_mask = df['bed_type'] == "Real Bed"

    df.loc[~real_bed_mask, 'bed_type'] = 'Other'

    return df


def group_rare_cancellation_policies(df):
    df['cancellation_policy'] = df['cancellation_policy'].replace({
        "super_strict_30": "strict",
        "super_strict_60": "strict"
    })

    return df

def group_rare_neighbourhoods(df):
    # TODO: Move top_neighbourhoods calculation to FeatureBuilder.fit()

    top_neighbourhoods = df['neighbourhood'] \
        .value_counts() \
        .head(TOP_NEIGHBOURHOODS) \
        .index
    
    df.loc[~df['neighbourhood'].isin(top_neighbourhoods),
           "neighbourhood"] = "Other Neighbourhood"

    return df

class AmenitiesPreprocessor:
    def __init__(self):
        self._encoder = MultiLabelBinarizer()
        

    def _prepare(self, df):
        df = self._create_amenities_list(df)
        df = self._remove_translation_missing_errors(df)
        df = self._clean_spelling_logical_errors(df)
        df = self._create_amenities_groups(df)

        return df

    def transform(self, df):
        if 'amenities_list' not in df.columns:
            df = self._prepare(df)

        amenities_encoded_df = pd.DataFrame(
            self._encoder.fit_transform(df['amenities_list']),
            columns=self._encoder.classes_,
            index=df.index
        )

        return amenities_encoded_df 
    
    def create_amenities_count(self, df):
        if 'amenities_list' not in df.columns:
            df = self._prepare(df)

        df['amenities_count'] = df['amenities_list'].str.len()

        return df

    def _create_amenities_list(self, df):
        df['amenities_list'] = df['amenities'].str.strip('{}') \
            .str.split(',') \
            .apply(lambda x: [item.strip().strip('"') for item in x])
        
        return df
        
    def _remove_translation_missing_errors(self, df):
        df['amenities_list'] = df['amenities_list'].apply(
            lambda x: [amenity for amenity in x if "translation missing" not in amenity]
        )

        return df
    
    def _clean_spelling_logical_errors(self, df):
        
        df['amenities_list'] = df['amenities_list'].apply(
            lambda amenities: [
                AMENITIES_REPLACEMENTS.get(amenity, amenity)
                for amenity in amenities
                if amenity not in AMENITIES_REMOVE
            ]
        )

        df['amenities_list'] = df['amenities_list'].apply(
            lambda x: [amenity for amenity in x if amenity != ""]
        )

        return df
    
    def _create_amenities_groups(self, df):
        def merge_amenities(amenities):
            amenities = set(amenities)

            for new_name, old_names in AMENITIES_GROUPS.items():
                if amenities & old_names:
                    amenities -= old_names
                    amenities.add(new_name)

            return sorted(amenities)
        
        df['amenities_list'] = df['amenities_list'].apply(merge_amenities)

        return df
    
class DescriptionPreprocessor:

    def transform(self, df, embedding_pca_components=None):
        df['description'] = df['description'].astype("string[python]")

        if EMBEDDINGS_PATH.exists():
            print(f"Loading embeddings from Parquet {EMBEDDINGS_PATH}...")

            embeddings_df = pd.read_parquet(EMBEDDINGS_PATH)
        
        else:
            print(f"Generating Embeddings using: {DEVICE}")
            
            df = self._clean_garbage(df)
            df = self._fix_whitespaces(df)
            
            model = SentenceTransformer("BAAI/bge-m3")
            embeddings = model.encode(
                df['description'].tolist(),
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=DEVICE
            )

            embeddings_df = pd.DataFrame(
                embeddings,
                columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
            )

            embeddings_df.to_parquet(EMBEDDINGS_PATH, index=False)

        if embedding_pca_components is not None:
            pca = PCA(n_components=embedding_pca_components, random_state=42)
            embeddings_pca = pca.fit_transform(embeddings_df)
            embeddings_df = pd.DataFrame(
                embeddings_pca,
                index=df.index,
                columns=[f"embedding_pca_{i}" for i in range(embeddings_pca.shape[1])]
            )
            print(f"Applying PCA ({embedding_pca_components} components)...")
            print(f"Embeddings shape is: {embeddings_df.shape}")

        return embeddings_df


    def _clean_garbage(self, df):
        mask = df['description'].isin(DESCRIPTION_GARBAGE_OBSERVATION)
        df.loc[mask, 'description'] = "No description"

        return df
    
    def _fix_whitespaces(self, df):
        df['description'] = df['description'] \
            .str.replace(r'\.{4,}', '...', regex=True) \
            .str.replace(r'([!?;:,])\1{2,}', 'r\1', regex=True)

        return df
    
def remove_columns(df):
    df.drop(columns=[
        "id",
        "thumbnail_url",
        "longitude",
        "latitude",
        "host_has_profile_pic",
        "name",
        "amenities",
        "description"
    ], inplace=True)

class FeatureBuilder:

    def get_df(self, df, use_amenities=False, use_embeddings=False, embedding_pca_components=None):
        if embedding_pca_components is not None and not use_embeddings:
            raise ValueError("embedding_pca_components can only be used when use_embeddings=True")

        df = df.copy()
        df = self._build_base_features(df)

        if use_amenities:
            amenities_df = self.get_amenities_df(df)
            df = df.join(amenities_df)
            df.drop(columns=['amenities_list'], inplace=True)

        if use_embeddings:
            embeddings_df = self.get_description_embeddings_df(df, embedding_pca_components)
            df = df.join(embeddings_df)

        remove_columns(df)

        return df

    def _build_base_features(self, df):

        # Fix boolean features
        df = fix_boolean_features(df)

        # Fill missing values
        df = fill_host_response_rate(df)
        df = fill_categorical_na_values(df)
        df = fill_numerical_na_values(df)

        # Create Datetime features and fill na values
        df = create_datetime_features(df)
        df = fill_datetime_na_values(df)

        # Fix beds issue, when beds == 0
        df = fix_beds(df)

        # Create a new feature based on longitude and latitude
        df = create_distance_to_listing_center(df)


        # Categorical features fix
        df = group_rare_bed_types(df)
        df = group_rare_cancellation_policies(df)
        df = group_rare_neighbourhoods(df)
        df = group_rare_property_types(df)
        
        return df
    
    def get_amenities_df(self, df):
        am_preprocessor = AmenitiesPreprocessor()
        amenities_df = am_preprocessor.transform(df)

        return amenities_df
    
    def get_description_embeddings_df(self, df, embedding_pca_components=None):
        embeddings_preprocessor = DescriptionPreprocessor()
        embeddings_df = embeddings_preprocessor.transform(df, embedding_pca_components)

        return embeddings_df


# def main():
#     df = pd.read_csv(DATASET_PATH)
#     df_copy = df.copy()

#     df_copy = fix_boolean_features(df_copy)

#     # Fill missing values
#     df_copy = fill_host_response_rate(df_copy)
#     df_copy = fill_categorical_na_values(df_copy)
#     df_copy = fill_numerical_na_values(df_copy)

#     # # Create Datetime features and fill na values
#     df_copy = create_datetime_features(df_copy)
#     df_copy = fill_datetime_na_values(df_copy)

#     # # Fix beds issue, when beds == 0
#     df_copy = fix_beds(df_copy)

#     # # Create a new feature based on latitude & longitude
#     df_copy = create_distance_to_listing_center(df_copy)

#     # # Categorical Features fixes
#     df_copy = group_rare_property_types(df_copy)
#     df_copy = group_rare_bed_types(df_copy)
#     df_copy = group_rare_cancellation_policies(df_copy)
#     df_copy = group_rare_neighbourhoods(df_copy)

#     # # Create amenities features and separate amenities encoded df
#     # amenities_preprocessor = AmenitiesPreprocessor()
#     # df_copy = amenities_preprocessor.create_amenities_count(df_copy)
#     # amenities_encoded_df = amenities_preprocessor.transform(df_copy)

#     # # Create description embedding DataFrame
#     # description_preprocessor = DescriptionPreprocessor()
#     # embeddings_df = description_preprocessor.transform(df_copy)

#     remove_columns(df_copy)

#     print(df_copy.isna().sum())

# if __name__ == "__main__":
#     main()