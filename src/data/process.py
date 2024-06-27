import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import Resampling
from rasterio.crs import CRS
from shapely.geometry import mapping
import xarray as xr
import pandas as pd
import datetime
import netCDF4 as nc
import re
import chardet
from datetime import datetime, timedelta
import os

def diff(serie):
  dif = serie.iloc[-1]-serie.iloc[0]
  return dif


def mmean(serie):
  m = np.mean(serie)
  return m


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


def num_to_ordinal(num):
  """
  Converts a number to its ordinal representation.

  Args:
      num: The number to be converted (must be an integer).

  Returns:
      str: The ordinal representation of the number (e.g., "1st", "2nd", "3rd", etc.).

  Raises:
      ValueError: If the input number is not an integer.
  """

  if not isinstance(num, int):
    raise ValueError("Input must be an integer")

  # Handle special cases for numbers ending in 11, 12, and 13
  if num % 100 in [11, 12, 13]:
    return str(num) + "th"

  # Get the last digit of the number
  last_digit = num % 10

  # Choose the appropriate suffix based on the last digit
  suffix = "st" if last_digit == 1 else ("nd" if last_digit == 2 else ("rd" if last_digit == 3 else "th"))

  return str(num) + suffix


def clip_data(np_data, shape_esm):
    """
    Clips the DataArray 'data' using the geometry from 'shape_esm'.

    Args:
        np_data (xr.DataArray): The DataArray to clip.
        shape_esm (geopandas.GeoDataFrame): GeoDataFrame containing the geometry to clip with.

    Returns:
        xr.DataArray: The clipped DataArray.
    """

    lat = np.arange(60, -60, -0.04)  # 3000 rows
    lon = np.arange(-180, 180, 0.04)  # 9000 cols

    data = xr.DataArray(data=np_data, dims=["lat", "lon"], coords=[lat, lon])
    data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    data.rio.write_crs("epsg:4326", inplace=True)

    data_esm = data.rio.clip(shape_esm.geometry.apply(mapping), shape_esm.crs, all_touched=True)
    lat = None
    lon = None
    data = None
    del lat
    del lon
    del data
    return data_esm 


def clip_array_with_geopackage(data, filename, crs=None):
  """
  Clips a NumPy array using a geopackage file.

  Args:
      data: The NumPy array to clip.
      filename: Path to the geopackage file.
      crs: Coordinate Reference System (CRS) of the data and shapefile (optional).

  Returns:
      A NumPy array representing the clipped data.
  """

  # Read the shapefile
  gdf = gpd.read_file(filename)

  # Get the first geometry (assuming only one shape is needed)
  geometry = mapping(gdf.iloc[0].geometry)

  # Open the raster dataset from the NumPy array
  with rasterio.open(None, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype) as src:
    src.transform = rasterio.Affine.identity  # Assuming unit transform for simplicity
    src.crs = crs if crs else CRS.from_epsg(4326)  # Default to EPSG:4326 (WGS84)
    src.write(data, 1)

    # Clip the raster by the geometry
    clipped, transform = rasterio.rasterize(shapes=[geometry], out_shape=data.shape, fill=0, transform=src.transform, crs=src.crs, resampling=Resampling.nearest)

  # Cleanup (optional, remove downloaded file after use)
  # import os
  # if os.path.exists(filename):
  #   os.remove(filename)

  return clipped


def create_dataframe_from_nc(interval, rela_path):
  
  """
  Creates a pandas DataFrame from a NetCDF .nc file with timestamps as columns.

  Args:
      interval (int): Time interval between timestamps (in hours).
      rela_path (str): Path to the NetCDF file.

  Returns:
      pandas.DataFrame: A 1D DataFrame containing timestamps as columns and data as rows.
  """

  with nc.Dataset(rela_path) as ds:
    base_time_str = ds.variables['datetime'].units
    parts = base_time_str.split("since")
    try:
      base_time = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M")
    except ValueError:
      base_time = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H")
    n_time = ds.variables['datetime'].shape[0]

    timestamps = [base_time + i * timedelta(hours=interval) for i in range(n_time)]
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


def get_latest_file_with_date(folder_path):
    latest_file = None
    latest_date = datetime.min

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_dvd.csv"):
                file_date_str = file[:8]
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                if file_date > latest_date:
                    latest_date = file_date
                    latest_file = os.path.join(root, file)
    latest_date = latest_date.date()
    return latest_file, latest_date
  

import pandas as pd
import re
from datetime import datetime

def read_esm_daily(csv_path, current_date):
  """
  Reads and preprocesses an ESM daily CSV file.

  Args:
      csv_path (str): Path to the CSV file.
      current_date (datetime.date): Current date object.

  Returns:
      pd.DataFrame: The preprocessed ESM daily DataFrame with datetime.date index.
  """
  with open(csv_path, 'rb') as f:
      rawdata = f.read()
      result = chardet.detect(rawdata)
      encoding = result['encoding']

  esm_daily = pd.read_csv(csv_path, encoding=encoding, header=None, skiprows=2)

  # Split each row by ';' and convert to DataFrame
  esm_daily = esm_daily[0].apply(lambda x: pd.Series(re.split(';', x)))
  esm_daily = esm_daily.T

  # Set column names and remove first row (containing column names)
  esm_daily.columns = esm_daily.iloc[0]
  esm_daily = esm_daily.iloc[6:]

  # Drop the last row and set index with 'N Sens' column (dropping it)
  esm_daily.drop(esm_daily.index[-1], inplace=True)
  esm_daily = esm_daily.set_index('N Sens', drop=True)

  # Convert time strings to datetime objects with specified year and date
  times = [datetime.strptime(time, '%H.%M') for time in esm_daily.index]
  datetimes = [t.replace(year=current_date.year, month=current_date.month, day=current_date.day) for t in times]
  esm_daily.index = datetimes
  try:
    esm_daily.columns = ['hidro_level_m1', 'precip_acumu_sm', 'hidro_level_sm']
  except ValueError:
    esm_daily.columns = ['hidro_level_m1']
    print(f"On the day of {current_date} only one sensor was working, {esm_daily.iloc[0].values}")
  return esm_daily




def make_esm_df(start_date: datetime, end_date: datetime, delimiter: str = "/") -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    
  """
  Reads ESM station data from CSV files within a date range.

  Args:
      start_date: The starting date for reading data (datetime object).
      end_date: The ending date for reading data (datetime object).
      delimiter: The delimiter used in the file paths (default: "/").

  Returns:
      A tuple containing two lists:
          - df_list_esm: A list of pandas DataFrames containing successfully read ESM data.
          - df_list_esm_err: A list of pandas DataFrames containing data with errors during reading.
  """
    
  list_esm: list[pd.DataFrame] = []

  for day in range((end_date - start_date).days + 1):
      current_date = start_date + timedelta(days=day)
      year = current_date.year
      month = format_number_with_zeros(current_date.month, 2)
      day = format_number_with_zeros(current_date.day, 2)
      csv_path = f"data{delimiter}raw{delimiter}Esm_Station{delimiter}yd{year}{delimiter}md{year}{month}{delimiter}{year}{month}{day}_dvd.csv"

      try:
          esm_daily = read_esm_daily(csv_path, current_date = current_date)

      except FileNotFoundError:
          print(f"File not found: {csv_path}")
          encoding = None
          df_temp = esm_daily.copy()
          df_temp.index = df_temp.index + pd.Timedelta(days=1)
          df_temp.iloc[:, :] = np.nan
          esm_daily = df_temp.copy()

      list_esm.append(esm_daily)
      if len(list_esm) > 1:
          df_esm = pd.concat(list_esm)
      else:
          df_esm = np.nan
      
  return df_esm


def process_esm_df(df_esm_all):
  """
  Preprocesses the provided DataFrame including storage type conversion and resampling.

  Args:
      df_esm_all (pd.DataFrame): The DataFrame containing ESM data.

  Returns:
      pd.DataFrame: The preprocessed DataFrame with a resampled frequency of 3 hours.
  """
  # Convert data to numeric (excluding existing numerics)
  df_esm_all = df_esm_all.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == "object" else x)

  # Convert specific columns to appropriate decimals
  conversion_dict = {
      'hidro_level_m1': 100,
      'hidro_level_sm': 100,
      'precip_acumu_sm': 1000
  }
  for col, divisor in conversion_dict.items():
      df_esm_all[col] = df_esm_all[col] / divisor

  # Replace -99.99 with NaN
  df_esm_all = df_esm_all.replace(-99.99, np.nan)

  # Resample data with 3-hour frequency (excluding NaNs)

  df_esm_3h = df_esm_all.resample('180min').agg({
      'hidro_level_m1': 'mean',
      'precip_acumu_sm': diff,  # Use custom diff function
      'hidro_level_sm': 'mean'
  })

  # Round values to 2 decimal places
  df_esm_3h = df_esm_3h.round(2)

  return df_esm_3h


import zlib
import os

def compress_file(input_file):
    """
    Compress a file using zlib compression and delete the original file.
    
    Args:
    input_file (str): Path to the file to be compressed.
    
    Returns:
    str: Full path of the compressed output file.
    """
    # Generate output file name by adding '_compressed.zlib' to the original filename
    base_path, ext = os.path.splitext(input_file)
    output_file = f"{base_path}_compressed.zlib"

    # Set compression parameters
    compression_level = 9  # Highest compression level (1-9)
    chunk_size = 8192  # Size of chunks to read/write (in bytes)

    # Open input and output files
    with open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            # Create a compressor object
            compressor = zlib.compressobj(compression_level)
            
            # Read, compress, and write data in chunks
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    # If no more data, flush the compressor and break the loop
                    f_out.write(compressor.flush())
                    break
                # Compress the chunk and write to output file
                compressed_chunk = compressor.compress(chunk)
                f_out.write(compressed_chunk)
    
    # Calculate file sizes in megabytes
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # Convert bytes to MB
    compressed_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert bytes to MB
    
    # Print compression results
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Original file size: {original_size:.2f} MB")
    print(f"Compressed file size: {compressed_size:.2f} MB")
    print(f"Compression ratio: {compressed_size / original_size:.2%}")

    # Delete the original input file
    os.remove(input_file)
    print(f"Input file {input_file} has been deleted.")

    # Get and print the full absolute path of the output file
    output_full_path = os.path.abspath(output_file)
    print(f"Compressed file saved at: {output_full_path}")

    # Return the full path of the compressed file
    return output_full_path