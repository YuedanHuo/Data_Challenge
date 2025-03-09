import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Function to create lag features BEFORE passing to the pipeline
def create_lag_features(X_df, n_lags=3):
    if 'Inflation_Rate' not in X_df.columns:
        raise ValueError("Missing 'Inflation_Rate' in dataset!")

    feature_array = X_df.copy()
    for lag in range(1, n_lags + 1):
        feature_array[f'inflation_lag_{lag}'] = feature_array['Inflation_Rate'].shift(lag)

    feature_array.drop(columns=['Inflation_Rate'], inplace=True)  # Drop target variable
    feature_array.dropna(inplace=True)  # Remove rows with NaN caused by shifting
    return feature_array

# Selected subset of variables for modeling
selected_features = [
    'Unemployment_Rate', 'GDP', 'Federal_Funds_Rate', 'Money_Supply', 'PPI',
    'SP500', 'DCOILWTICO', 'DGS10'
]

# Column Transformer with feature selection using indices
def get_feature_transformer(feature_count):
    return ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), list(range(feature_count)))  # Scale all features
        ],
        remainder='passthrough'
    )

# Final Model Pipeline
def get_estimator():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
        ('feature_transform', get_feature_transformer(len(selected_features) + 3)),  # Adjust for lag features
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])


