import datetime
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import IPython


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