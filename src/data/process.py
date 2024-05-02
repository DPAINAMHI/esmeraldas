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