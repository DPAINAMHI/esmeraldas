from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from src.models.score import nse, kge


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


