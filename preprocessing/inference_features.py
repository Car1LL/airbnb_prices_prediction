import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "Airbnb_Data.csv"
EMBEDDINGS_PATH = ROOT / "dataset" / "description-embeddings_01.parquet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOOLEAN_FIX = {
    "instant_bookable",
    "host_identity_verified"
}

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

EARTH_RADIUS = 6371
TOP_NEIGHBOURHOODS = 209

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

class InferenceFeatureBuilder:

    def __init__(self, use_amenities=False, use_embeddings=False, embedding_pca_components=None):
        if embedding_pca_components is not None and not use_embeddings:
            raise ValueError(
                "embedding_pca_components can only be used "
                "when use_embeddings=True"
            )

        self.use_amenities=use_amenities
        self.use_embeddings=use_embeddings

        # Parameters learned during fit
        self.host_response_rate_median = None
        self.numerical_medians = {}
        self.datetime_medians = {}

        self.host_reference_date = None
        self.last_review_reference_date = None

        self.listing_centers = None
        self.rare_property_types = set()
        self.top_neighbourhoods = None

        # Specialized preprocessors
        self.amenities_preprocessor = AmenitiesPreprocessor() if use_amenities else None
        self.description_preprocessor = (
            DescriptionPreprocessor(embedding_pca_components=embedding_pca_components)
            if use_embeddings else None
        )

    # ==================== FIT ====================
    def fit(self, df):
        df = df.copy()

        # Learn parameters required for base preprocessing
        self._fit_host_response_rate(df)
        self._fit_numerical_na_values(df)
        self._fit_datetime_features(df)
        self._fit_datetime_na_values(df)
        self._fit_listing_centers(df)
        self._fit_rare_property_types(df)
        self._fit_neighbourhoods(df)

        # Amenities encoder
        if self.use_amenities:
            self.amenities_preprocessor.fit(df)

        # Description embeddings + PCA
        if self.use_embeddings:
            self.description_preprocessor.fit(df)

        return self

    # ==================== TRANSFORM ====================

    def transform(self, df):
        df = df.copy()

        # Base features
        df = self._transform_base_features(df)

        # Amenities
        if self.use_amenities:
            amenities_df = self.amenities_preprocessor.transform(df)

            df = df.join(amenities_df)

            if "amenities_list" in df.columns:
                df.drop(columns=['amenities_list'], inplace=True)

        # Description embeddings
        if self.use_embeddings:
            embeddings_df = self.description_preprocessor.transform(df)

            df = df.join(embeddings_df)

        self.remove_columns(df)

        return df

    # ==================== BASE TRANSFORM ====================

    def _transform_base_features(self, df):

        # Fix boolean types
        df = self._transform_boolean_features(df)

        # Fix missing values
        df = self._transform_host_response_rate(df)
        df = self._fill_categorical_na_values(df)
        df = self._transform_numerical_na_values(df)

        # Datetime features
        df = self._transform_datetime_features(df)
        df = self._transform_datetime_na_values(df)

        # Fix beds
        df = self._transform_beds(df)

        # Distance to listing center
        df = self._transform_distance_to_listing_center(df)

        # Categorical features
        df = self._transform_bed_types(df)
        df = self._transform_cancellation_policies(df)
        df = self._transform_neighbourhoods(df)
        df = self._transform_rare_property_types(df)

        return df

    # ==================== FIT METHODS ====================

    def _fit_host_response_rate(self, df):
        df['host_response_rate'] = pd.to_numeric(
            df['host_response_rate'].str.replace("%", "", regex=False),
            errors="coerce"
        )

        self.host_response_rate_median = df['host_response_rate'].median()

    def _fit_numerical_na_values(self, df):
        numerical_features = df.select_dtypes(include=['float', 'int']).columns

        excluded_columns = {
            "host_response_rate",
        }

        for column in numerical_features:
            if column in excluded_columns:
                continue

            self.numerical_medians[column] = df[column].median()

    def _fit_datetime_features(self, df):
        self.host_reference_date = pd.to_datetime(
            df['host_since'], errors="coerce"
        ).max()

        self.last_review_reference_date = pd.to_datetime(
            df['last_review'], errors="coerce"
        ).max()

    def _fit_datetime_na_values(self, df):
        datetime_features = [
            "host_age_days",
            "host_since_year",
            "review_history_days",
            "days_since_last_review",
            "first_review_year"
        ]

        temp_df = df.copy()
        temp_df = self._transform_datetime_features(temp_df)

        for col in datetime_features:
            self.datetime_medians[col] = temp_df[col].median()

    def _fit_listing_centers(self, df):
        self.listing_centers = df.groupby("city")[["latitude", "longitude"]] \
            .median() \
            .rename(columns={
                "latitude": "listing_center_lat",
                "longitude": "listing_center_lon"
            })

    def _fit_rare_property_types(self, df):
        property_counts = df['property_type'].value_counts()

        self.rare_property_types = set(
            property_counts[property_counts < 10].index
        )

    def _fit_neighbourhoods(self, df):
        self.top_neighbourhoods = df['neighbourhood'].value_counts() \
            .head(TOP_NEIGHBOURHOODS) \
            .index

    # ==================== TRANSFORM METHODS ====================

    def _transform_boolean_features(self, df):
        mappings = {
            "t": True,
            "f": False
        }

        for col in BOOLEAN_FIX:
            df[col] = df[col].map(mappings)
            df[col] = df[col].fillna(-1).astype("int8")

        boolean_features = df.select_dtypes(include=['bool']).columns
        df[boolean_features] = df[boolean_features].astype('int8')

        return df
    
    def _transform_host_response_rate(self, df):
        df['host_response_rate'] = pd.to_numeric(
            df['host_response_rate'].str.replace("%", "", regex=False),
            errors="coerce"
        )

        df['host_response_rate_missing'] = df['host_response_rate'].isna() \
            .astype("int8")

        df['host_response_rate'] = df['host_response_rate'].fillna(self.host_response_rate_median)

        return df
    
    def _fill_categorical_na_values(self, df):
        categorical_features = df.select_dtypes(include=['string', 'object']).columns

        excluded_columns = {
            "host_response_rate",
            "host_since",
            "last_review",
            "first_review"
        }

        for column in categorical_features:
            if column in excluded_columns:
                continue

            df[column] = df[column].fillna("Unknown ")

        return df
    
    def _transform_numerical_na_values(self, df):
        for column, median in self.numerical_medians.items():
            df[column] = df[column].fillna(median)

        return df

    def _transform_datetime_features(self, df):
        date_features = {
            "last_review",
            "first_review",
            "host_since"
        }

        for col in date_features:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        df['host_age_days'] = (self.host_reference_date - df['last_review']).dt.days
        df['days_since_last_review'] = (self.last_review_reference_date - df['last_review']).dt.days
        df['review_history_days'] = (df['last_review'] - df['first_review']).dt.days

        df['first_review_year'] = df['first_review'].dt.year
        df['host_since_year'] = df['host_since'].dt.year

        df['has_reviews'] = df['first_review'].notna().astype("int8")
        df['review_before_host'] = (df['first_review'] < df['host_since']).fillna(False).astype("int8")

        df.drop(
            columns=[
                "first_review",
                "last_review",
                "host_since"
            ], inplace=True
        )

        return df

    def _transform_datetime_na_values(self, df):
        for column, median in self.datetime_medians.items():
            df[column] = df[column].fillna(median)

        return df

    def _transform_beds(self, df):
        invalid_beds = df['beds'] == 0
        single_guest = df['accommodates'] == 1
        multiple_guests = df['accommodates'] > 1

        df.loc[invalid_beds & single_guest, 'beds'] = 1
        df.loc[invalid_beds & multiple_guests, 'beds'] = (
            df.loc[invalid_beds & multiple_guests, 'accommodates'] - 1
        )

        return df

    def _transform_distance_to_listing_center(self, df):
        df = df.merge(
            self.listing_centers,
            on='city',
            how='left'
        )

        lat1 = np.radians(df['latitude'])
        lon1 = np.radians(df['longitude'])

        lat2 = np.radians(df['listing_center_lat'])
        lon2 = np.radians(df['listing_center_lon'])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np. sin(dlat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        c = 2 * np.arcsin(np.sqrt(a))

        df['distance_to_listing_center'] = EARTH_RADIUS * c
        df.drop(columns=[
            "listing_center_lon",
            "listing_center_lat"
        ], inplace=True)

        return df

    def _transform_bed_types(self, df):
        real_bed_mask = df['bed_type'] == "Real Bed"
        df.loc[~real_bed_mask, "bed_type"] = "Other"

        return df
    
    def _transform_rare_property_types(self, df):
        df['property_type'] = df['property_type'].replace(
            self.rare_property_types,
            "Other"
        )

        return df

    def _transform_cancellation_policies(self, df):
        df['cancellation_policy'] = df['cancellation_policy'].replace({
            "super_strict_30": "strict",
            "super_strict_60": "strict"
        })

        return df

    def _transform_neighbourhoods(self, df):
        df.loc[
            ~df['neighbourhood'].isin(self.top_neighbourhoods),
            "neighbourhood"
        ] = "Other Neighbourhood"

        return df

    def remove_columns(self, df):
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

class AmenitiesPreprocessor:

    def __init__(self):
        self._encoder = MultiLabelBinarizer()

    def fit(self, df):
        df = self._prepare(df)

        self._encoder.fit(df['amenities_list']) 

        return self

    def transform(self, df):
        df = self._prepare(df)

        amenities_encoded_df = pd.DataFrame(
            self._encoder.transform(df['amenities_list']),
            columns=self._encoder.classes_,
            index=df.index
        )

        return amenities_encoded_df

    def create_amenities_count(self, df):
        if "amenities_list" not in df.columns:
            df = self._prepare(df)

        df['amenities_count'] = df['amenities_list'].str.len()

        return df

    def _prepare(self, df):
        df = self._create_amenities_list(df)
        df = self._remove_translation_missing_errors(df)
        df = self._clean_spelling_logical_errors(df)
        df = self._create_amenities_groups(df)

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

    def __init__(self, embedding_pca_components=None):
        self.embedding_pca_components = embedding_pca_components
        self._pca = None
        self._embedding_model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)

    def fit(self, df):
        embeddings_df = self._create_embeddings(df)

        if self.embedding_pca_components is not None:
            self._pca = PCA(
                n_components=self.embedding_pca_components,
                random_state=42
            )

            self._pca.fit(embeddings_df)

        return self

    def transform(self, df):
        embeddings_df = self._create_embeddings(df)

        if self._pca is not None:
            embeddings_pca = self._pca.transform(embeddings_df)

            embeddings_df = pd.DataFrame(
                embeddings_pca,
                index=df.index,
                columns=[f"embedding_pca_{i}" for i in range(embeddings_pca.shape[1])]
            )

        return embeddings_df

    def _create_embeddings(self, df):
        df = df.copy()

        if EMBEDDINGS_PATH.exists():
            print(f"Loading embedding parquet file from: {EMBEDDINGS_PATH}")
            embeddings_df = pd.read_parquet(EMBEDDINGS_PATH)

            if df.index.isin(embeddings_df.index).all():
                return embeddings_df.loc[df.index]

        else:
            embeddings_df = None

        if embeddings_df is not None:
            missing_mask = ~df.index.isin(embeddings_df.index)
            missing_df = df.loc[missing_mask]
        else:
            missing_df = df

        print(f"Generating embeddings for: {len(missing_df)} descriptions...")
        missing_df['description'] = missing_df['description'].astype("string[python]")
        missing_df = self._clean_garbage(missing_df)
        missing_df = self._fix_whitespaces(missing_df)

        embeddings = self._embedding_model.encode(
            missing_df['description'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=DEVICE
        )

        new_embeddings_df = pd.DataFrame(
            embeddings,
            index=missing_df.index,
            columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
        )

        if embeddings_df is not None:
            embeddings_df = pd.concat([
                embeddings_df, new_embeddings_df
            ])

        else:
            embeddings_df = new_embeddings_df

        embeddings_df.to_parquet(EMBEDDINGS_PATH, index=True)

        return embeddings_df.loc[df.index]

    def _clean_garbage(self, df):
        mask = df['description'].isin(DESCRIPTION_GARBAGE_OBSERVATION)
        df.loc[mask, 'description'] = "No description"

        return df
    
    def _fix_whitespaces(self, df):
        df['description'] = df['description'] \
            .str.replace(r'\.{4,}', '...', regex=True) \
            .str.replace(r'([!?;:,])\1{2,}', 'r\1', regex=True)

        return df