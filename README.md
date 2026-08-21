# Introduction

The project is end-to-end machine learning solution for **predicting Airbnb listing prices** based on listing characteristics such as location, property type, room type, amenities, reviews, host information and listing description. 

The project uses the [Airbnb Open dataset](https://www.kaggle.com/datasets/rupindersinghrana/airbnb-price-dataset) dataset containing **74,111** listings. The target variable is `log_price`, making this a regression problem. The model predicts the logarithm of the listing price during training, with predictions transformed back to the original dollar scale during inference.

The dataset was obtained from Kaggle:

* Dataset: [Airbnb Dataset](https://www.kaggle.com/datasets/rupindersinghrana/airbnb-price-dataset)
* Task: **Regression**
* Target: `log_price`
* Number of records: **74,111**

The dataset is distributed under the **Apache License**

The main goal of the project was not only to train an accurate regression model, but to build a complete and reproducible ML workflow covering data preprocessing, feature engineering, text embeddings, dimensionality reduction, model training, evaluation, artifact persistence and inference through a **FastAPI** service.

# Exploratory Data Analysis

The EDA phase includes a comprehensive analysis of the dataset, including data structure, feature distribution, missing values and relationships between features and the target variable.

Alongside the analysis of the original features, a small analysis of **engineered features** was performed. This approach made it possible to evaluate both, the original and newly created features within a single notebook, saving time during the exploratory phase.

To simplify feature engineering, the project introduced the `FeatureBuilder` class located at:

`./preprocessing/features.py`

The class allows creating a transformed copy of the original `pandas.DataFrame` with engineered features in a single step, making it convenient to experiments with different feature representations during the research phase. 

Detailed EDA results and visualizations can be found in:

`./notebooks/00_EDA.ipynb`


>   NOTE: Although most of the model research notebooks initially used `FeatureBuilder`, it was later discovered that its implementation introduced **data leakage** into the training pipeline. The class was subsequently replaced by `InferenceFeatureBuilder`, located at `./preprocessing/inference_features.py`. This implementation separates the `fit` and `transform` stages and ensures that statistics and transformations are learned only from the training data.

>   The data leakage issue and the process of identifying and fixing it are discussed in more detail in the **Data Leakage Section**

# Model Training & Experimentation

Model experimentation was conducted across three notebooks:

* `./notebooks/01_Linear_Modeling.ipynb`;
* `./notebooks/02_Tree_Based_Modeling.ipynb`;
* `./notebooks/03_CatBoost.ipynb`;

These notebooks contain detailed experiments, configurations and evaluation results for each group models. Rather than duplicating the full experimentation process in this *README*, only the main approaches and observations are summarized here.

All trained models and their corresponding artifacts from the experiments were saved to:

`./artifacts/`

Most of the models were implemented using their native API. Since **XGBoost** required additional functionality for saving and restoring both and trained model and its configuration, a custom pipeline was implemented in:

`./utils/xgb_pipeline.py`

The `XGBoostPipeline` class encapsulates the **XGBoost** model together with its parameters, allowing the complete model configuration to be persisted and restored during inference.

> NOTE: During the data leakage investigation, the **XGBoost** pipeline was slightly modified to ensure > that preprocessing performed during inference does not depend on information from the test or > training datasets. The updated implementation is available at:
>
> ```python
> ./utils/inference_xgb_pipeline.py
> ```
> 
> The class inherits from the original XGBoost pipeline and provides functionality required for leakage-free inference.

All computationally intensive model experiments used **Optuna** for hyperparameter optimization.

## Key Features Engineering Approaches

### Amenities 

The original `amenities` feature was stored as a string representing a list of amenities. It was parsed into individual amenities, with each unique amenity transformed into a binary feature indicating whether it was present in a listing.

### Text Embeddings

The `description` feature was transformed into a **1024-dimensional embedding vector** representing the semantic information contained in the listing description.

Sentence embeddings were generated using the `sentence-transformers` library with the `BAAI/bge-m3` model.

A more detailed description of the embedding generation process can be found in:

`./notebooks/00_EDA.ipynb`

### Dimensionality Reduction

The generated sentence embeddings were further analyzed to determine an appropriate number of principal components Based on the explained variance analysis, **370 components** were selected as a suitable representation of the original embeddings.

The embeddings vectors were therefore compressed from **1024 to 370 dimensions** using PCA (`sklearn.decomposition.PCA`), significantly reducing the dimensionality of the text feature space while preserving majority of its useful information.


# Data Leakage

The **data leakage** issue was discovered at a relatively late stage of the project, while preparing the model and preprocessing pipeline for deployment through **FastAPI**.

At this point, **XGBoost** had been selected as the best-performing model based on the model comparison experiments documented in:

`./notebooks/04_Model_Comparison.ipynb`

However, while preparing the inference pipeline, it was discovered that feature engineering logic was being fitted on the **entire dataset**, including both training and test subsets. This meant that information from the test set could indirectly influence the features used for model training, resulting in data leakage.

The main source of this issue was the `FeatureBuilder` class:

`./preprocessing/features.py`

Since the feature engineering logic was centralized in this class, a dedicated analysis notebook was created to investigate and verify the problem:

`./notebooks/Data_Leakage_Analysis.ipynb`

## Fix

To eliminate the leakage, a new `InferenceFeatureBuilder` class was introduced:

`./preprocessing/inference_features.py`

The new implementation separates the feature engineering process into two distinct stages:

* `fit()` - learns all required statistics and parameters **only from the training data**;
* `transform()` - applies the previously learned transformations to new data without fitting anything again;

This separations ensures that information from the validation or test data cannot be used during feature engineering and provides a consistent preprocessing pipeline for both training and inference.

After replacing `FeatureBuilder` with `InferenceFeatureBuilder`, the data leakage was successfully eliminated. However, the model's evaluation metrics **decreased compared to the original experiments**. This was expected, as the previous results had been artificially improved by information leaking from the test set into a feature engineering process.

A detailed investigation, comparison of the leakage and leakage-free pipelines, and the resulting metrics can be found in:

`./notebooks/Data_Leakage_Analysis.ipynb`

# Final Model

The initial choice for the model was **XGBoost**, based on the model comparison experiments. However, after identifying and eliminating the data leakage described in the previous section, the final model choice was reconsidered.

The project ultimately uses **CatBoost** at the final model regression model.

The main reasons for choosing CatBoost were:

* **Simple workflow** - the model can be trained and used without introducing an additional custom optimization pipeline;
* **Fast training** - the final model can be trained in a reasonable amount of time without requiring extensive hyperparameter optimization;
* **Competitive performance** - after removing data leakage, CatBoost demonstrated performance comparable to the other tree-based approaches;
* **Simple production pipeline** - using CatBoost allowed the final workflow to remain relatively straightforward and easy to maintain;

The detailed comparison and reasoning behind the final model selection can be found in:

`./notebooks/Data_Leakage_Analysis.ipynb`

## Inference Logic

The inference and model management logic was separated from FastAPI application and placed in:

`./src/`

The main module responsible for model lifecycle is:

`./src/model.py`

It contains logic required to:

1. Check whether trained model and preprocessing artifacts already exist;
2. Train the model if the required artifacts are missing;
3. Save the trained CatBoost model;
4. Save the fitted `InferenceFeatureBuilder` together with the parameters learned during training;
5. Ensure that the same fitted preprocessing logic is used during inference through `transform()`;

This separation keeps the model lifecycle and preprocessing logic independent from the FastAPI application itself, making the inference pipeline easier to test, maintain and reuse.


# Evaluation

The final CatBoost model was evaluated on the held-out test set after applying the leakage-free `InferenceFeatureBuilder` pipeline:


| Metric | Test | Train |
| :---: | :---: | :---: | 
| **MAE** | $50.02 | $44.83 |
| **RMSE** | $114.22 | $100.28 |
| **R2 Score** | 0.71 | 0.77 |

The final model achieves a Test MAE of **$50.02**, meaning that the model's predictions deviate from the actual listing price by approximately $50 on average.

These metrics represent the final **leakage-free** performance of the model and should be considered the reference results for the project.

# Installation

    git clone https://github.com/Car1LL/airbnb_prices_prediction.git
    cd airbnb_prices_prediction
    pip install -r requirements.txt

## Dataset Setup

1. Download dataset from [here](https://www.kaggle.com/datasets/rupindersinghrana/airbnb-price-dataset)
2. Unzip the downloaded archive
3. Place `Airbnb_Data.csv` into the `./dataset/` directory.

Make sure the project structure matches the following:

## Project structure

    .
    ├── app
    │   ├── main.py
    │   └── schemas.py
    ├── dataset
    │   └── Airbnb_Data.csv
    ├── notebooks
    │   ├── 00_EDA.ipynb
    │   ├── 01_Linear_Modeling.ipynb
    │   ├── 02_Tree_Based_Modeling.ipynb
    │   ├── 03_CatBoost.ipynb
    │   ├── 04_Model_Comparison.ipynb
    │   └── Data_Leakage_Analysis.ipynb
    ├── preprocessing
    │   ├── features.py
    │   ├── inference_features.py
    │   ├── linear_preprocessor.py
    │   └── tree_preprocessor.py
    ├── README.md
    ├── requirements.txt
    ├── src
    │   ├── model.py
    │   └── prediction.py
    └── utils
        ├── inference_xgb_pipeline.py
        └── xgb_pipeline.py

The `./artifacts/` directory is created automatically after training and contains the trained model and preprocessing artifacts.

# Jupyter Notebooks

To inspect EDA and Modeling notebooks, in terminal:

    cd ./notebooks/
    jupyter lab

# Training model

    python -m src.model

This will train the model and save artifacts into the `./artifacts/` directory.

# FastAPI Service

Open the root directory and run in terminal:

    uvicorn app.main:app --reload

Then open a browser and follow the link: http://127.0.0.1:8000/docs