import pandas as pd
import numpy as np
import os

def stat_ana_features(arr):
    # Get the count of non-NaN values
    non_nan_count = np.count_nonzero(~np.isnan(arr))
    print(f"Count (non-NaN): {non_nan_count}")  # Count (non-NaN): 9

    # Get the sum of non-NaN values
    non_nan_sum = np.nansum(arr)
    print(f"Sum (non-NaN): {non_nan_sum}")  # Sum (non-NaN): 45.0

    # Get the mean of non-NaN values
    non_nan_mean = np.nanmean(arr)
    print(f"Mean (non-NaN): {non_nan_mean}")  # Mean (non-NaN): 5.0

    # Get the median of non-NaN values
    non_nan_median = np.nanmedian(arr)
    print(f"Median (non-NaN): {non_nan_median}")  # Median (non-NaN): 5.0

    # Get the standard deviation of non-NaN values
    non_nan_std = np.nanstd(arr)
    print(f"Standard Deviation (non-NaN): {non_nan_std}")  # Standard Deviation (non-NaN): 2.7386127875258306

    # Get the minimum and maximum values (ignoring NaN)
    non_nan_min = np.nanmin(arr)
    non_nan_max = np.nanmax(arr)
    print(f"Minimum (non-NaN): {non_nan_min}")  # Minimum (non-NaN): 1.0
    print(f"Maximum (non-NaN): {non_nan_max}")  # Maximum (non-NaN): 9.0

    # Get the quartile values (ignoring NaN)
    # Remove NaN and 0 values from the array
    arr = arr[~np.isnan(arr) & (arr != 0)]
    quartiles = np.nanquantile(arr, [0.25, 0.5, 0.75])
    print(f"Quartiles (non-NaN and non-0): {quartiles}")  # Quartiles (non-NaN): [3.0, 5.0, 7.0]

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


def keep_5th_and_12th(group):
    # Get the rows from 5th to 12th
    filtered_group = group.iloc[4:12, :]
    
    # Get the index of the filtered group
    index = filtered_group.index
    
    # Create a new index with modified timestamps
    new_index = []
    for i, timestamp in enumerate(index):
        if i >= 5:  # If the row index is >= 9 (10th row and beyond)
            new_index.append(timestamp + pd.Timedelta(days=1))
        else:
            new_index.append(timestamp)
    
    # Update the index of the filtered group
    filtered_group.index = new_index
    
    return filtered_group

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#
#——————————————————————————————————————————— Get features for the data from ESM sensor station ————————————————————————————————————————#
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

from ftplib import FTP
import os
import datetime
import ftplib
from math import floor
from src.model.features import create_lagged_features
from src.data.process import process_esm_df, read_esm_daily


def download_esm_single_day(ftp_server, username, password, target_date):
  """
  Downloads ESM station data from an FTP server for a specified date.

  Args:
      ftp_server (str): The FTP server address.
      username (str): The username for FTP access.
      password (str): The password for FTP access.
      target_date (datetime.date): The date for which to download data.

  Returns:
      str: The path to the downloaded local file (if downloaded), None otherwise.
  """

  local_file_path = None

  # Connect to the FTP server
  ftp = FTP(ftp_server)
  ftp.login(username, password)

  # Change to the desired directory
  ftp.cwd("TEST_ESTACIONES_AUTOMATICAS/H5033/D1/F10")

  # Check if target date is a valid date object
  if not isinstance(target_date, datetime.date):
    print(f"Invalid target_date format. Please provide a datetime.date object.")
    return None

  # Extract year and month from target date
  target_year = target_date.year
  target_month = target_date.month

  try:
    # Iterate through folders based on target date
    for yd_folder in [f"yd{target_year}"]:
      # Change to the "yd" folder (assuming single year folder)
      ftp.cwd(yd_folder)

      for md_folder in [f"md{target_year}{target_month:02d}"]:
        # Change to the "md" folder for the target month
        ftp.cwd(md_folder)

        # List all CSV files
        csv_files = [file for file in ftp.nlst() if file.endswith(".csv")]

        for csv_file in csv_files:
          # Extract the date from the CSV file name
          file_date = datetime.datetime.strptime(csv_file[:8], "%Y%m%d").date()

          # Check if the file date matches the target date
          if file_date == target_date:
            # Download the file
            with open(csv_file, "wb") as local_file:
              ftp.retrbinary(f"RETR {csv_file}", local_file.write)
            local_file_path = os.path.abspath(local_file.name)
            print(f"Downloaded file: {local_file_path}")

        # Go back to the parent directory (md folder)
        ftp.cwd("..")

      # Go back to the parent directory (yd folder)
      ftp.cwd("..")

  except ftplib.all_errors as e:
    # Handle FTP errors including potentially missing folders
    print(f"Error encountered while accessing folders for {target_date}: {e}")

  ftp.quit()

  return local_file_path


def is_last_row_timestamp_valid(df):
  """
  Checks if the timestamp of the last row in a DataFrame is greater than an adjacent hourly time less than the current time and a multiple of 3.

  Args:
      df (pd.DataFrame): The DataFrame with a datetime index.

  Returns:
      bool: True if the last row timestamp is valid, False otherwise.
  """

  # Get the current time
  current_time = datetime.datetime.now()

  # Calculate the expected adjacent hourly time (multiple of 3)
  expected_hour = floor(current_time.hour / 3) * 3
  expected_datetime = datetime.datetime.combine(datetime.date.today(), datetime.time(expected_hour, 0))

  # Get the timestamp of the last row (assuming datetime index)
  last_row_timestamp = df.index[-1]

  # Check if the last row timestamp is greater than the expected time
  is_valid = last_row_timestamp > expected_datetime

  # Print informative message
  print(f"Current time: {current_time.strftime('%H:%M')}")
  print(f"Expected time (multiple of 3 less than current hour): {expected_datetime.strftime('%H:%M')}")
  print(f"Last row timestamp: {last_row_timestamp.strftime('%H:%M')}")
  d_time = current_time - expected_datetime
  d_min = d_time.total_seconds()/60
  if is_valid:
      print("Last row timestamp is valid (data received within the expected 3-hour window).")
  elif d_min < 30:
      print(f"Last row timestamp is not valid, please wait for at least {30 - d_min:.0f} minutes.")
  else:
      print("Last row timestamp is not valid, please check the data source.")

  return is_valid


def get_esm_features(ftp_server, username, password):
  """
  Downloads ESM sensor data for specified dates, processes them into a DataFrame,
  and creates lagged features (if applicable).

  Args:
      ftp_server (str): The FTP server address.
      username (str): The username for FTP access.
      password (str): The password for FTP access.
      download_dates (list[datetime.date]): A list of dates for which to download data.

  Returns:
      pd.DataFrame (optional): The processed DataFrame containing combined ESM data
                                 for all download dates (if processed successfully).
                                 None otherwise.
  """
  today = datetime.date.today()
  yesterday = today - datetime.timedelta(days=1)
  download_dates = [yesterday, today]
  df_esm_2d = []
  for download_date in download_dates:
    file_path = download_esm_single_day(ftp_server, username, password, download_date)

    # Check if file was downloaded successfully (consider error handling)
    if not file_path:
      print(f"Failed to download data for {download_date}")
      continue

    df_esm_daily = read_esm_daily(file_path, download_date)
    df_esm_2d.append(df_esm_daily)

    # Delete downloaded file after processing
    os.remove(file_path)
    print(f"File deleted successfully: {file_path}")

  if not df_esm_2d:
    print("No data downloaded for any of the specified dates.")
    return None

  df_esm_2d = pd.concat(df_esm_2d)

  # Data validation (e.g., check for valid timestamps)
  if not is_last_row_timestamp_valid(df_esm_2d):
    print("Invalid timestamp in the last row. Data processing skipped.")
    return None

  df_esm_3h = process_esm_df(df_esm_2d)
  df_esm_3h = create_lagged_features(df_esm_3h, 'hidro_level_sm', 5)
  newest_features = df_esm_3h.iloc[-1,:]
  return newest_features


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#
#——————————————————————————————————————————— Get features for the data from CHRS data portal ——————————————————————————————————————————#
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

from urllib.parse import urlparse
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from affine import Affine
from pyramids.dataset import Dataset
import requests
import gzip
import struct
from datetime import datetime as dt
from src.model.predict import *
import re
from datetime import timedelta


def make_url(interval):
  url_base = "https://persiann.eng.uci.edu/CHRSdata/PERSIANN-CCS/"
  now = dt.now()

  # how to determine the hours based on the updating scheme of the data source
  # current updating scheme: time stamp of t (UTC+0) will be available at t+1.5 (UTC+0), where t is multiples of 3, and t-5 (UTC-5)
  # is the local target time, covering t-5 to t-2 (UTC-5). then the available time would be t-3.5 (UTC-5)
  # hence given a local current time h, 
  base_hour_local = get_adjacent_multiple_of_three_plus_one(now.hour)
  if now.hour < 1:
     base_time_local = dt(now.year, now.month, now.day-1, base_hour_local, 0, 0)
  else:
     base_time_local = dt(now.year, now.month, now.day, base_hour_local, 0, 0)
  time_difference = now - base_time_local
  if time_difference.total_seconds() <= 0*60: # 90*60: 90mins apart
    print("Data source has not updated yet.")
    return None
  if base_hour_local >=21:
     local_hour_to_predict =1
  else:
     local_hour_to_predict = base_hour_local+3
  hh_utc = local_hour_to_predict +2 -3
  if hh_utc >= 24:
     hh_utc = hh_utc -24
     now = now + dt.timedelta(days=1)
  hh_formatted = format_number_with_zeros(hh_utc, 2)  # Hour with leading zeros
  freq = str(interval) + 'h'

  doy = now.timetuple().tm_yday  # Day of the year
  year_2d = str(now.year)[-2:]  # Last two digits of the year
  doy_formatted = format_number_with_zeros(doy, 3)

  # Construct the URL
  url = url_base + str(interval) + "hrly/" + "rgccs" + freq + year_2d + doy_formatted + hh_formatted + '.bin.gz'
  print(f'The local base time is {base_time_local}\n The local hour to predict is {local_hour_to_predict}')
  print(f'The ccs file used for prediction is from {url}')
  return url


import os
from urllib.parse import urlparse

def download_ccs_file(url):
    """
    Download a file from a given URL and save it locally if it doesn't exist.

    Args:
        url (str): The URL of the file to be downloaded.
        local_dir (str): The local directory where the file should be saved.

    Returns:
        str or None: The full path of the downloaded or existing file if successful, None otherwise.
    """

    try:
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # Construct the full path for the local file
        local_filename = os.path.join(os.getcwd(), filename)

        # Check if the file already exists in the local directory
        if os.path.isfile(local_filename):
            print(f'File already exists: {local_filename}')
            return local_filename

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Get the content of the file
            file_content = response.content

            # Open a local file for writing in binary mode
            with open(local_filename, 'wb') as file:
                # Write the downloaded content to the local file
                file.write(file_content)

            print(f'File downloaded successfully: {local_filename}')
            return local_filename
        else:
            print(f'Failed to download file. Error code: {response.status_code}')
            return None

    except requests.exceptions.RequestException as e:
        print(f'Failed to download file. An error occurred: {e}')
        return None


def convert_bin_gz_to_tif(input_filepath, output_filepath='output.tif', ncol=9000, nrow=3000, upper_left_x=0.0, upper_left_y=60.0, pixel_width=0.04, pixel_height=-0.04, nodata_value=-9999, crs='EPSG:4326'):
    """
    Converts a compressed binary file (.bin.gz) to a GeoTIFF file (.tif), and adjusts its coordinate so that it matches the CRS of the shapefile.

    Args:
        input_filepath (str): Path to the input compressed binary file (.bin.gz).
        output_filepath (str, optional): Path to the output GeoTIFF file (.tif). Default is 'output.tif'.
        ncol (int, optional): Number of columns in the raster. Default is 9000.
        nrow (int, optional): Number of rows in the raster. Default is 3000.
        upper_left_x (float, optional): X coordinate of the upper-left corner of the raster. Default is 0.0.
        upper_left_y (float, optional): Y coordinate of the upper-left corner of the raster. Default is 60.0.
        pixel_width (float, optional): Width of each pixel. Default is 0.04.
        pixel_height (float, optional): Height of each pixel. Default is -0.04.
        nodata_value (float or int, optional): Value representing no data. Default is -9999.
        crs (str or rasterio.crs.CRS, optional): Coordinate reference system of the raster. Default is 'EPSG:4326'.

    Returns:
        None
    """
    # Decompress the binary data
    with gzip.open(input_filepath, 'rb') as f_in:
        decompressed_data = f_in.read()

    # Determine the data type of the binary data
    data_type = np.dtype('>i2')

    # Create a NumPy array from the decompressed data
    binary_array = np.frombuffer(decompressed_data, data_type)
    binary_array = binary_array.reshape(nrow, ncol)

    # Convert the data type to float32 and replace nodata values with np.nan
    binary_array = binary_array.astype(np.float32)
    binary_array[binary_array == nodata_value] = np.nan

    # Apply unit conversion (mm/3hr * 100)
    binary_array *= 0.01  # Convert from (mm/3hr * 100) to mm/3hr
    binary_array = np.around(binary_array, decimals=2)

    transform = Affine.from_gdal(upper_left_x, pixel_width, 0, upper_left_y, 0, pixel_height)

    # Create the GeoTIFF file
    with rasterio.open(output_filepath, 'w', driver='GTiff', height=nrow, width=ncol, count=1, dtype=binary_array.dtype, nodata=np.nan, crs=crs, transform=transform) as dst:
        dst.write(binary_array, 1)

    tif = Dataset.read_file(output_filepath)

    # how to release the usage of the tif object?
    # so that it could be deleted from the disk cache later
    new_tif = tif.convert_longitude()

    # release the memory of tif to avoid conflict of the same output_filepath with the new_tif
    del tif
    new_tif.to_file(output_filepath)


def clip_raster(full_raster_path, mask_path, output_path):
  """
  Clips a raster .tif file using a vector file as a mask.

  Args:
      raster_path (str): Path to the input raster file (e.g., .tif).
      mask_path (str): Path to the vector file used as a mask (e.g., .gpkg).
      output_path (str): Path to save the clipped raster file.

  Returns:
      None
  """

  # Open the raster file
  with rasterio.open(full_raster_path) as src:
      # Read the vector file (without using with)
      gdf = gpd.read_file(mask_path, driver='GPKG')

      if len(gdf.geometry) == 1:
          geom = gdf.geometry.values[0]
      else:
          raise ValueError("Multiple geometries found in the mask file. Please provide a single geometry.")

      # Clip the raster using the geometry
      out_image, out_transform = mask(src, [geom], crop=True)

      # Copy the metadata
      out_meta = src.meta.copy()

      # Update the metadata with the new transform and dimensions
      out_meta.update({"driver": "GTiff",
                      "height": out_image.shape[1],
                      "width": out_image.shape[2],
                      "transform": out_transform})

      # Write the clipped raster to a new file
      with rasterio.open(output_path, "w", **out_meta) as dest:
          dest.write(out_image)


def flatten_and_filter_features(array_clipped_rectangular):
  """
  Input the 2d original rectangular array of the clipped pixels and returns a 1d flat array of the features.
  """
  clipped_1d = array_clipped_rectangular.ravel()
  clipped_features = clipped_1d[~np.isnan(clipped_1d)]
  return clipped_features





def extract_datetime_from_url(url):
    # Extract the filename from the URL
    filename = url.split('/')[-1]
    
    # Use regex to extract the datetime part
    match = re.search(r'rgccs3h(\d{2})(\d{3})(\d{2})\.bin\.gz', filename)
    
    if match:
        year, day_of_year, hour = match.groups()
        
        # Convert to full year (assuming 20xx)
        full_year = int(f"20{year}")
        
        # Create a datetime object
        date = dt(full_year, 1, 1) + timedelta(days=int(day_of_year) - 1, hours=int(hour))
        
        return date
    else:
        return None




def get_ccs_features(shape_file_path, url=None):
    """
    Retrieve and process CCS (Climate Change Scenario) features for a given shape file.

    This function downloads CCS data, converts it to a GeoTIFF format, clips it to the area 
    defined by the input shape file, and returns the processed data as a flattened and filtered array.

    Args:
        shape_file_path (str): Path to the shape file defining the area of interest.
        url (str, optional): URL to download the CCS data. If None, a URL is generated
                             for a 3-day interval. Defaults to None.

    Returns:
        numpy.ndarray or None: A 1D array of CCS features for the defined area. 
                               Returns None if URL generation fails.

    Raises:
        Various exceptions may be raised by the called functions, including:
        - URLError: If there's an issue downloading the file.
        - IOError: If there are issues reading or writing files.
        - RuntimeError: If GDAL operations fail.

    Side Effects:
        - Downloads a .bin.gz file temporarily.
        - Creates and then deletes temporary .tif files.

    Note:
        This function relies on several helper functions that are not defined here:
        make_url(), download_ccs_file(), convert_bin_gz_to_tif(), clip_raster(),
        and flatten_and_filter_features().
    """
    if url is None:
        url = make_url(interval=3)
        if url is None:
            return None
    
    # Download the .bin.gz file given the generated url
    downloaded_file_path = download_ccs_file(url)
    date = extract_datetime_from_url(url)
    formatted_datetime = date.strftime("%Y_%m_%d_%H_%M_%S")
    # Set up the parameters to convert the .bin.gz file to .tif file
    world_tif_filepath = f'output_{formatted_datetime}.tif'

    # Convert the downloaded ccs .bin.gz file to .tif file and correct its CRS
    raster_array = convert_bin_gz_to_tif(downloaded_file_path, world_tif_filepath)  
    
    # Clip the adjusted raster with the shapefile
    clipped_tif_path = f'output_clipped_{formatted_datetime}.tif'
    clip_raster(world_tif_filepath, shape_file_path, clipped_tif_path)

    # Read the clipped raster into a numpy array of features
    with rasterio.open(clipped_tif_path) as src:
        clipped_array_rectangular = src.read(1)
    
    # Flatten and filter the original rectangular array
    ccs_feature_array = flatten_and_filter_features(clipped_array_rectangular)

    # Delete all the temporary local files
    os.remove(downloaded_file_path)  # Delete the downloaded .bin.gz file
    os.remove(world_tif_filepath)  # Delete the converted original .tif raster
    # os.remove(clipped_tif_path)  # Delete the clipped raster
    
    return ccs_feature_array


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#
#——————————————————————————————————————————— Get features for the data from wrf data ——————————————————————————————————————————#
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#


