def create_lagged_features_original(df, column_name, max_lag):
    for lag in range(1, max_lag + 1):
        df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
    return df

def create_lagged_features(df, column_name, max_lag):
  """
  This function adds lagged features to a Pandas DataFrame.

  Args:
    df (pandas.DataFrame): The DataFrame to add lagged features to.
    column_name (str): The name of the column to create lagged features for.
    max_lag (int): The maximum lag to create. Lags will be created from 1 to max_lag.

  Returns:
    pandas.DataFrame: The DataFrame with the new lagged features added.
  """

  # Loop through the desired lags (1 to max_lag)
  for lag in range(1, max_lag + 1):
    # Create a new column name with the format '{column_name}_lag_{lag}'
    new_column_name = f'{column_name}_lag_{lag}'

    # Add a new column to the DataFrame with the lagged values of the original column
    df[new_column_name] = df[column_name].shift(lag)

  # Return the DataFrame with the new lagged features
  return df
