import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import TimeSeriesSplit

problem_title = 'Forecasting Inflation Using Expanded Macroeconomic Data'
_target_column_name = 'Inflation_Rate'
_ignore_column_names = []

# Define regression prediction type (for continuous forecasting)
Predictions = rw.prediction_types.make_regression()

# Define workflow
workflow = rw.workflows.Estimator()

# Define score types
score_types = [
    rw.score_types.RMSE(name='rmse', precision=4),
    rw.score_types.MARE(name='mae', precision=4),
]

# Time-based cross-validation
N_SPLITS = 5  # Number of time-based splits
def get_cv(X, y):
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    return cv.split(X, y)

# READ DATA
def _read_data(path, filename):
    df = pd.read_csv(os.path.join(path, 'data', filename), index_col=0, parse_dates=True)
    y_array = df[_target_column_name].values.astype(float)  # Inflation target
    X_df = df.drop(columns=[_target_column_name])  # Features
    return X_df, y_array

# Get train & test data
def get_train_data(path='.'): 
    return _read_data(path, 'train_expanded.csv')

def get_test_data(path='.'): 
    return _read_data(path, 'test_expanded.csv')

 
