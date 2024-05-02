import datetime
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import IPython
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

def create_time_series_dataframe(interval, rela_path):
  """
  Creates a pandas DataFrame from a NetCDF file with timestamps as columns.

  Args:
      interval (int): Time interval between timestamps (in hours).
      rela_path (str): Path to the NetCDF file.

  Returns:
      pandas.DataFrame: A 1D DataFrame containing timestamps as columns and data as rows.
  """

  with nc.Dataset(rela_path) as ds:
    base_time_str = ds.variables['datetime'].units
    parts = base_time_str.split("since")
    base_time = datetime.datetime.strptime(parts[1].strip(), "%Y-%m-%d %H")
    n_time = ds.variables['datetime'].shape[0]

    timestamps = [base_time + i * datetime.timedelta(hours=interval) for i in range(n_time)]
    data = ds.variables['precip'][:]

    # Reshape data directly for efficiency
    data_reshaped = data.reshape([n_time, -1])

    # Create DataFrame with timestamps as rows and data as rows
    df = pd.DataFrame(data_reshaped, index=timestamps)

  return df

def iterate_months(start_year, start_month, end_year, end_month):
  """
  Iterates through months from a start date to an end date, returning formatted strings (YYYYMM).

  Args:
      start_year (int): Starting year (e.g., 2019).
      start_month (int): Starting month (1-12).
      end_year (int): Ending year (e.g., 2024).
      end_month (int): Ending month (1-12).

  Yields:
      str: Formatted month string (YYYYMM).
  """

  current_year = start_year
  current_month = start_month

  while (current_year, current_month) <= (end_year, end_month):
    formatted_month = f"{current_year:04d}{current_month:02d}"
    yield formatted_month  # Use yield for iteration

    if current_month == 12:
      current_month = 1
      current_year += 1
    else:
      current_month += 1

def format_number_with_zeros(number, desired_digits):
  """Formats a number with leading zeros to reach the desired number of digits.

  Args:
      number: The integer to format.
      desired_digits: The desired number of digits in the output string.

  Returns:
      A string representation of the number with leading zeros if needed.
  """

  if not isinstance(number, int) or desired_digits <= 0:
    raise ValueError("Invalid input: number must be an integer and desired_digits must be positive.")

  # Convert the number to a string
  number_str = str(number)

  # Add leading zeros if needed
  num_leading_zeros = desired_digits - len(number_str)
  formatted_string = "0" * num_leading_zeros + number_str

  return formatted_string

def diff(serie):
  dif = serie.iloc[-1]-serie.iloc[0]
  return dif
def mmean(serie):
  m = np.mean(serie)
  return m

# Function to calculate Nash-Sutcliffe Efficiency
def nse(observed, simulated):
    return 1 - sum((simulated-observed)**2)/sum((observed-np.mean(observed))**2)

# Function to calculate Kling-Gupta Efficiency
def kge(observed, simulated):
    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.std(simulated) / np.std(observed)
    beta = np.sum(simulated) / np.sum(observed)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def create_lagged_features(df, column_name, max_lag):
    for lag in range(1, max_lag + 1):
        df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
    return df

def prepare_data(df_merged, train_size, column_y):
    """
    Prepare and split data into training and testing sets.

    Args:
        df_merged (pandas.DataFrame): The merged DataFrame containing features and target variable.
        train_size (float): The proportion of data to be used for training (between 0 and 1).
        column_y (str): The name of the column containing the target variable.

    Returns:
        tuple: A tuple containing the following:
            X_train (pandas.DataFrame): The training data features.
            X_test (pandas.DataFrame): The testing data features.
            y_train (pandas.Series): The training data target variable.
            y_test (pandas.Series): The testing data target variable.
            dates_train (pandas.DatetimeIndex): The training data dates.
            dates_test (pandas.DatetimeIndex): The testing data dates.
    """
    # Drop rows with NaN values
    df_merged = df_merged.dropna()
    print(f"Shape of the final dataset is {df_merged.shape}\n")

    # Preparing data
    X = df_merged.drop(columns=column_y)
    y = df_merged[column_y]
    dates = df_merged.index

    # Splitting data (train_size for training, 1 - train_size for testing)
    split_index = int(len(X) * train_size)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    dates_train, dates_test = dates[:split_index], dates[split_index:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, 
                           n_estimators=250, max_depth=20, min_samples_split=5, 
                           min_samples_leaf=5, max_features='sqrt', bootstrap=True, n_jobs=-2):
  """
  Trains a Random Forest Regressor model, makes predictions, and calculates evaluation metrics.

  Args:
      X_train: Training features.
      y_train: Training target variable.
      X_test: Testing features.
      y_test: Testing target variable.
      n_estimators: Number of trees in the forest (default: 250).
      max_depth: Maximum depth of individual trees (default: 20).
      min_samples_split: Minimum number of samples required to split a node (default: 5).
      min_samples_leaf: Minimum number of samples required to be at a leaf node (default: 5).
      max_features: Number of features to consider at each split (default: 'sqrt').
      bootstrap: Use bootstrap sampling when building trees (default: True).
      n_jobs: Number of CPUs to use during training (-1 uses all CPUs) (default: -2).

  Returns:
      nse_train: Nash-Sutcliffe Efficiency on training data.
      nse_test: Nash-Sutcliffe Efficiency on testing data.
      mse_train: Mean Squared Error on training data.
      mse_test: Mean Squared Error on testing data.
      kge_train: Kling-Gupta Efficiency on training data. (Assuming function kge is defined)
      kge_test: Kling-Gupta Efficiency on testing data. (Assuming function kge is defined)
      model:
      y_train_pred: 
      y_test_pred:
  """

  # Train the Random Forest model
  model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs)
  model.fit(X_train, y_train)

  # Make predictions on training and testing data
  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)

  # Assuming functions for nse, mse, and kge are defined elsewhere
  nse_train = nse(y_train, y_train_pred)
  nse_test = nse(y_test, y_test_pred)
  mse_train = mean_squared_error(y_train, y_train_pred)
  mse_test = mean_squared_error(y_test, y_test_pred)
  kge_train = kge(y_train, y_train_pred)
  kge_test = kge(y_test, y_test_pred)

  # Print the evaluation metrics
  print(f"Training NSE: {nse_train}, MSE: {mse_train}, KGE: {kge_train}")
  print(f"Testing NSE: {nse_test}, MSE: {mse_test}, KGE: {kge_test}")

  # Return the calculated metrics
  return nse_train, nse_test, mse_train, mse_test, kge_train, kge_test, model, y_train_pred, y_test_pred

def grid_search(X_train, y_train, X_test, y_test):
    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [10, 15, 20, 30],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10, 15]
    }

    # Create a random forest regressor
    rf_regressor = RandomForestRegressor(bootstrap=True, n_jobs=-2, random_state=42, max_features='sqrt')

    # Perform grid search with custom scoring function
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring=make_scorer(nse, greater_is_better=True))
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters:", best_params)

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Predicting discharge
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculating metrics
    nse_train = nse(y_train, y_train_pred)
    nse_test = nse(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    kge_train = kge(y_train, y_train_pred)
    kge_test = kge(y_test, y_test_pred)

    # Outputting the metrics
    print(f"Training NSE: {nse_train}, MSE: {mse_train}, KGE: {kge_train}")
    print(f"Testing NSE: {nse_test}, MSE: {mse_test}, KGE: {kge_test}")

    return best_model, y_train_pred, y_test_pred

def plot_fitting_result(y_train, y_train_pred, dates_train, y_test, y_test_pred, 
                         dates_test, nse_train, mse_train, kge_train, nse_test, mse_test, kge_test, X_train):
    # Create a DataFrame for training data
    train_data = pd.DataFrame({
        'Observed': y_train,
        'Simulated': y_train_pred,
        'Date': dates_train
    })

    # Create a DataFrame for testing data
    test_data = pd.DataFrame({
        'Observed': y_test,
        'Simulated': y_test_pred,
        'Date': dates_test
    })

    # Create a figure for training data
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Observed'], mode='lines', name='Observed'))
    fig_train.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Simulated'], mode='lines', name='Simulated'))
    fig_train.update_layout(
        title=f"Training Period: Observed vs. Simulated Discharge<br>NSE: {nse_train:.2f}, MSE: {mse_train:.2f}, KGE: {kge_train:.2f}, Number of Features: {X_train.shape[1]}",
        xaxis_title='Date',
        yaxis_title='Discharge',
        legend=dict(x=0.1, y=1.1, orientation='h')
    )
    fig_train.update_xaxes(tickangle=45)

    # Create a figure for testing data
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Observed'], mode='lines', name='Observed'))
    fig_test.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Simulated'], mode='lines', name='Simulated'))
    fig_test.update_layout(
        title=f"Testing Period: Observed vs. Simulated Discharge<br>NSE: {nse_test:.2f}, MSE: {mse_test:.2f}, KGE: {kge_test:.2f}, Number of Features: {X_train.shape[1]}",
        xaxis_title='Date',
        yaxis_title='Discharge',
        legend=dict(x=0.1, y=1.1, orientation='h')
    )
    fig_test.update_xaxes(tickangle=45)

    # Show the plots
    fig_train.show()
    fig_test.show()

def plot_feature_importance(X_train, model):
    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame with feature names and importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    })

    # Sort the DataFrame by importance in descending order
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Create the feature importance plot
    fig = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance', color='importance')

    # Customize the layout
    fig.update_layout(
        xaxis_tickangle=-45,  # Rotate x-axis labels
        xaxis_title='Feature',
        yaxis_title='Importance',
        coloraxis_colorbar=dict(title='Importance')
    )

    # Show the plot
    fig.show()
    return feature_importance
