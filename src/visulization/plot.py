import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def plot_fitting_result(y_train, y_train_pred, dates_train, y_test, y_test_pred, 
                       dates_test, nse_train, mse_train, kge_train, nse_test, mse_test, kge_test, X_train):
  """
  This function creates and displays time series plots comparing observed vs. simulated discharge 
  for both training and testing periods, along with performance metrics.

  Args:
    y_train (numpy.array): Observed discharge values for training data.
    y_train_pred (numpy.array): Simulated discharge values for training data.
    dates_train (list): List of dates corresponding to training data.
    y_test (numpy.array): Observed discharge values for testing data.
    y_test_pred (numpy.array): Simulated discharge values for testing data.
    dates_test (list): List of dates corresponding to testing data.
    nse_train (float): Nash-Sutcliffe Efficiency (NSE) for training data.
    mse_train (float): Mean Squared Error (MSE) for training data.
    kge_train (float): Kling-Gupta Efficiency (KGE) for training data.
    nse_test (float): Nash-Sutcliffe Efficiency (NSE) for testing data.
    mse_test (float): Mean Squared Error (MSE) for testing data.
    kge_test (float): Kling-Gupta Efficiency (KGE) for testing data.
    X_train (numpy.array): Training features used for model fitting.
  """

  # Prepare DataFrames for plotting
  train_data = pd.DataFrame({
    'Observed': y_train,
    'Simulated': y_train_pred,
    'Date': dates_train
  })

  test_data = pd.DataFrame({
    'Observed': y_test,
    'Simulated': y_test_pred,
    'Date': dates_test
  })

  # Create plot for training data
  fig_train = go.Figure()
  fig_train.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Observed'], mode='lines', name='Observed'))
  fig_train.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Simulated'], mode='lines', name='Simulated'))
  fig_train.update_layout(
    title=f"Training Period: Observed vs. Simulated Discharge\nNSE: {nse_train:.2f}, MSE: {mse_train:.2f}, KGE: {kge_train:.2f}, Features: {X_train.shape[1]}",
    xaxis_title='Date',
    yaxis_title='Discharge',
    legend=dict(x=0.1, y=1.1, orientation='h')
  )
  fig_train.update_xaxes(tickangle=45)

  # Create plot for testing data (similar structure to training data plot)
  fig_test = go.Figure()
  # ... (similar code as training data plot)
  fig_test.update_layout(
    title=f"Testing Period: Observed vs. Simulated Discharge\nNSE: {nse_test:.2f}, MSE: {mse_test:.2f}, KGE: {kge_test:.2f}, Features: {X_train.shape[1]}",
    xaxis_title='Date',
    yaxis_title='Discharge',
    legend=dict(x=0.1, y=1.1, orientation='h')
  )
  fig_test.update_xaxes(tickangle=45)

  # Display the plots
  fig_train.show()
  fig_test.show()


def plot_feature_importance(X_train, model):
  """
  This function plots the feature importances for a fitted model.

  Args:
    X_train (pandas.DataFrame): The training features used to fit the model.
    model (object): The fitted model object.

  Returns:
    pandas.DataFrame: A DataFrame containing feature names and their corresponding importances.
  """

  # Extract feature importances from the model
  importances = model.feature_importances_

  # Create a DataFrame for storing feature names and importances
  feature_importance = pd.DataFrame({
    'feature': X_train.columns,  # Column names from training features
    'importance': importances
  })

  # Sort DataFrame by feature importance in descending order
  feature_importance = feature_importance.sort_values('importance', ascending=False)

  # Create a bar chart to visualize feature importance
  fig = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance', color='importance')

  # Customize the plot layout for better readability
  fig.update_layout(
    xaxis_tickangle=-45,           # Rotate x-axis labels to prevent overlapping
    xaxis_title='Feature',
    yaxis_title='Importance',
    coloraxis_colorbar=dict(title='Importance')  # Label for colorbar
  )

  # Display the plot
  fig.show()

  # Additionally, return the DataFrame containing feature importances
  return feature_importance




















def plot_feature_importance_original(X_train, model):
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


def plot_fitting_result_original(y_train, y_train_pred, dates_train, y_test, y_test_pred, 
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