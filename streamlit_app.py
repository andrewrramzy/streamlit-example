import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

# Import streamlit and pycaret
import streamlit as st
from pycaret.classification import *

# Load the data
st.title("Pycaret Streamlit Interface")
st.write("This is a simple interface for using Pycaret functions with streamlit.")
data = st.file_uploader("Upload your data file here (csv format only)", type="csv")
if data is not None:
    df = pd.read_csv(data)
    st.dataframe(df)

    # Set up the experiment
    st.sidebar.title("Setup Parameters")
    st.sidebar.write("Choose the parameters for the setup function")

    # Get the list of categorical and numeric features
    cat_features = st.sidebar.multiselect("Select the categorical features", df.columns)
    num_features = [col for col in df.columns if col not in cat_features]

    # Get the target column
    target = st.sidebar.selectbox("Select the target column", df.columns)

    # Get the train size
    train_size = st.sidebar.slider("Select the train size (fraction)", 0.0, 1.0, 0.7)

    # Get the sampling parameters
    sampling = st.sidebar.checkbox("Enable sampling")
    sampling_method = st.sidebar.selectbox("Select the sampling method", ["Cluster Centroids", "Random Over Sampler", "Random Under Sampler", "NearMiss", "Neighbourhood Cleaning Rule", "SMOTE", "SMOTETomek", "SMOTENC", "TomekLinks"])
    sampling_strategy = st.sidebar.selectbox("Select the sampling strategy", ["minority", "majority", "not minority", "not majority", "all", "auto"])

    # Get the feature engineering parameters
    feature_selection = st.sidebar.checkbox("Enable feature selection")
    feature_selection_threshold = st.sidebar.slider("Select the feature selection threshold", 0.0, 1.0, 0.8)
    feature_interaction = st.sidebar.checkbox("Enable feature interaction")
    feature_ratio = st.sidebar.checkbox("Enable feature ratio")
    interaction_threshold = st.sidebar.slider("Select the interaction threshold", 0.0, 1.0, 0.01)

    # Get the transformation parameters
    normalize = st.sidebar.checkbox("Enable normalization")
    normalize_method = st.sidebar.selectbox("Select the normalization method", ["zscore", "minmax", "maxabs", "robust"])
    transformation = st.sidebar.checkbox("Enable transformation")
    transformation_method = st.sidebar.selectbox("Select the transformation method", ["yeo-johnson", "quantile"])

    # Get the PCA parameters
    pca = st.sidebar.checkbox("Enable PCA")
    pca_method = st.sidebar.selectbox("Select the PCA method", ["linear", "kernel", "incremental"])
    pca_components = st.sidebar.number_input("Enter the number of PCA components", 1, len(df.columns)-1, 5)

    # Get the ignore features
    ignore_features = st.sidebar.multiselect("Select the features to ignore", df.columns)

    # Get the silent and verbose parameters
    silent = st.sidebar.checkbox("Enable silent mode")
    verbose = st.sidebar.checkbox("Enable verbose mode")

    # Run the setup function
    st.write("Running the setup function...")
    exp = setup(data=df, 
                target=target, 
                categorical_features=cat_features, 
                numeric_features=num_features, 
                train_size=train_size, 
                sampling=sampling, 
                sampling_method=sampling_method, 
                sampling_strategy=sampling_strategy, 
                feature_selection=feature_selection, 
                feature_selection_threshold=feature_selection_threshold, 
                feature_interaction=feature_interaction, 
                feature_ratio=feature_ratio, 
                interaction_threshold=interaction_threshold, 
                normalize=normalize, 
                normalize_method=normalize_method, 
                transformation=transformation, 
                transformation_method=transformation_method, 
                pca=pca, 
                pca_method=pca_method, 
                pca_components=pca_components, 
                ignore_features=ignore_features, 
                silent=silent, 
                verbose=verbose)
    st.write("Setup completed.")

    # Add the compare model function
    st.title("Compare Models")
    st.write("This function compares the performance of different models using various metrics.")
    include = st.multiselect("Select the models to include", ["lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "catboost"])
    exclude = st.multiselect("Select the models to exclude", ["lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "catboost"])
    sort = st.selectbox("Select the metric to sort the results", ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"])
    n_select = st.number_input("Enter the number of best models to select", 1, len(include)-len(exclude), 1)
    turbo = st.checkbox("Enable turbo mode")
    cross_validation = st.checkbox("Enable cross validation")
    fold = st.number_input("Enter the number of folds for cross validation", 2, 10, 10)
    round = st.number_input("Enter the number of decimal places to round the results", 1, 6, 4)
    st.write("Running the compare model function...")
    best_models = compare_models(include=include, exclude=exclude, sort=sort, n_select=n_select, turbo=turbo, cross_validation=cross_validation, fold=fold, round=round)
    st.write("Compare model function completed.")
