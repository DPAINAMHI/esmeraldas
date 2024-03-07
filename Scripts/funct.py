import datetime
import netCDF4 as nc
import numpy as np
import pandas as pd


def create_time_series_dataframe(interval, rela_path):
  """
  Creates a pandas DataFrame from a NetCDF file with timestamps as columns.

  Args:
      interval (int): Time interval between timestamps (in hours).
      rela_path (str): Path to the NetCDF file.

  Returns:
      pandas.DataFrame: A DataFrame containing timestamps as columns and data as rows.
  """

  with nc.Dataset(rela_path) as ds:
    base_time_str = ds.variables['datetime'].units
    parts = base_time_str.split("since")
    base_time = datetime.datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M")
    n_time = ds.variables['datetime'].shape[0]

    timestamps = [base_time + i * datetime.timedelta(hours=interval) for i in range(n_time)]
    data = ds.variables['precip'][:]

    # Reshape data directly for efficiency
    data_reshaped = data.reshape([n_time, -1])

    # Create DataFrame with timestamps as rows and data as rows
    df = pd.DataFrame(data_reshaped, index=timestamps)

  return df


