import os
import pandas as pd
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.warp import Resampling
from rasterio.crs import CRS
from shapely.geometry import mapping
from rasterio.transform import from_origin
from rasterio.mask import mask
import urllib.request
from datetime import date
from datetime import timedelta
import xarray as xr
import rioxarray
from tqdm import tqdm
import requests
import math

def read_persiann_css_online0(url, nrow, ncol, dtype=np.int16):
    """Downloads, decompresses a Persiann CCS .bin.gz file, converts it to a NumPy array,
        sets -9999 values to NaN, divides all values by 100, and reshapes to 2x2.

        Args:
            url: The URL of the Persiann CCS .bin.gz file.
            dtype: The desired data type for the NumPy array (default: np.float32).

        Returns:
            A reshaped NumPy array containing the processed data as a nrow*ncol matrix.

        Raises:
            URLError: If an error occurs while downloading the file.
            ValueError: If the decompressed data size is not compatible with nrow*ncol reshape.
    """
    # Try opening the URL and decompressing the data
    try:
        compressed_data = requests.get(url).content
        decompressed_data = gzip.decompress(compressed_data)
        # Convert to NumPy array
        data = np.frombuffer(decompressed_data, dtype=np.dtype('>h')).astype(float) 
        data = data.reshape((nrow,ncol))
        data_1 = data[:,int(ncol/2):]
        data_2 = data[:,:int(ncol/2)]
        data = np.hstack((data_1,data_2))
        data= data/100
        data[data < 0] = np.nan
        data = np.flipud(data)
        print(f"Data successfully downloaded from {url}")
        compressed_data = None
        decompressed_data = None
        data_1 = None
        data_2 = None
        del compressed_data
        del decompressed_data
        del data_1
        del data_2
        return data
    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"Error reading file from {url}: {e}")


def read_persiann_css_online(url, nrow, ncol):
    """Downloads, decompresses a Persiann CCS .bin.gz file, converts it to a NumPy array,
       sets -9999 values to NaN, divides all values by 100, and reshapes to 2x2.
       With visualized progress bar while downloading each file

    Args:
        url: The URL of the Persiann CCS .bin.gz file.
        nrow: Number of rows for reshaping the data.
        ncol: Number of columns for reshaping the data.
        dtype: The desired data type for the NumPy array (default: np.float32).

    Returns:
        A reshaped NumPy array containing the processed data as a nrow*ncol matrix.

    Raises:
        URLError: If an error occurs while downloading the file.
        ValueError: If the decompressed data size is not compatible with nrow*ncol reshape.
    """

    try:
        # Open the URL, handle unknown content length
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', None))  # Get total size (if available)
            progress_bar = tqdm(total=total_size, desc="Downloading data from "+url, unit='B', unit_scale=True, unit_divisor=1024) if total_size is not None else tqdm(desc="Downloading data")

            # Alternative 1: Read data in chunks using read(chunksize)
            compressed_data = b''
            chunksize = 1024*100 # update per 100 kb of downloading
            while True:
                chunk = response.read(chunksize)
                if not chunk:
                    break
                compressed_data += chunk
                if total_size is not None:  # Update progress if total size known
                    progress_bar.update(len(chunk))

            # Alternative 2: Read entire data (if content length is known or for small files)
            # if total_size is not None:
            #     compressed_data = response.read()

            progress_bar.close()  # Close progress bar after download

        # Decompress data
        decompressed_data = gzip.decompress(compressed_data)
        data = np.frombuffer(decompressed_data, dtype=np.dtype('>h')).astype(float) 
        data = data.reshape((nrow,ncol))
        data_1 = data[:,int(ncol/2):]
        data_2 = data[:,:int(ncol/2)]
        data = np.hstack((data_1,data_2))
        data= data/100
        data = np.round(data,2)
        data[data < 0] = np.nan
        data = np.flipud(data)
        return data
    
    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"Error reading file from {url}: {e}")

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

def iter_url(start_year, start_month, start_day, end_year, end_month, end_day, interval, max_num_of_obs_per_slice):
    """
    Generates a set of URLs and corresponding date ranges based on the given parameters.

    Args:
        start_year: The starting year of the date range.
        start_month: The starting month of the date range.
        start_day: The starting day of the date range.
        end_year: The ending year of the date range.
        end_month: The ending month of the date range.
        end_day: The ending day of the date range.
        interval: The interval between URLs in hours (e.g., 3 for every 3 hours).
        max_num_of_obs_per_slice: The maximum number of observations per data slice.

    Returns:
        A list of dictionaries, each containing:
        - urls: A list of generated URLs for the slice.
        - date_slice: A pandas date range object representing the slice.
        - start_datetime (pd.Timestamp): The start datetime of the slice.
        - end_datetime (pd.Timestamp): The end datetime of the slice.
        The number of observations in the given time range combining all slices
    """

    # Define the start and end time points
    start_time = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_time = pd.Timestamp(year=end_year, month=end_month, day=end_day, hour=23,minute=59,second=59)
    if start_time>end_time:
       raise ValueError("start date must be no more recent than the end date inputed")
    # Construct the time range with the specified interval and inclusive "left" boundary
    freq = str(interval) + 'h'
    time_range = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive='left')
    ranges = []
    url_base = "https://persiann.eng.uci.edu/CHRSdata/PERSIANN-CCS/"

    # Iterate over time slices
    num_of_slice=math.ceil(len(time_range)/max_num_of_obs_per_slice)
    for i in range(num_of_slice):
        slice_start_index = i * max_num_of_obs_per_slice
        slice_end_index = min((i + 1) * max_num_of_obs_per_slice - 1, len(time_range) - 1)  # Handle cases where the last slice might be shorter
        slice_start = time_range[slice_start_index]
        slice_end = time_range[slice_end_index]
        date_slice = time_range[slice_start_index:slice_end_index + 1]  # Use direct slicing for DatetimeIndex
        urls = []
        # Iterate over all days within the slice
        for time_point in date_slice:
            doy = time_point.timetuple().tm_yday  # Day of the year
            year_2d = str(time_point.year)[-2:]  # Last two digits of the year
            doy_formatted = format_number_with_zeros(doy, 3)
            hh_formatted = format_number_with_zeros(time_point.hour, 2)  # Hour with leading zeros
            # Construct the URL
            url = url_base + str(interval) + "hrly/" + "rgccs" + freq + year_2d + doy_formatted + hh_formatted + '.bin.gz'
            urls.append(url)

        range_dict = {
            "urls": urls,
            "date_slice": date_slice,
            "start_datetime": slice_start,
            "end_datetime": slice_end,
        }
        ranges.append(range_dict)

    print(f"{len(time_range)} datasets from {start_time.date()} to {end_time.date()} with a frequency of {freq} will be downloaded from {url_base}")
    return ranges, len(time_range)

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

def download_geopackage(url, filename="temp.gpkg"):
  """
  Downloads a geopackage file from a URL.

  Args:
      url: The URL of the online geopackage file.
      filename: Temporary filename to store the downloaded geopackage (optional).

  Returns:
      The gpd.read_file() of the downloaded geopackage.
  """
  # Download the geopackage file
  response = requests.get(url, stream=True)
  if response.status_code == 200:
    with open(filename, 'wb') as f:
      for chunk in response.iter_content(1024):
        f.write(chunk)
    print(f"Succeeded in downloading geopackage from {url}")
    shape_file=gpd.read_file(filename)
    del response
    return shape_file
  else:
    raise ValueError(f"Failed to download geopackage from {url}")

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

def read_persiann_ccs(file_path):
    ncols, nrows = 9000, 3000
    data = np.zeros((nrows, ncols), dtype=np.float32)  # Initialize data array

    with gzip.open(file_path, 'rb') as f:
        for i in range(nrows):
            for j in range(ncols):
                # Read two bytes from the file, big-endian format
                val = struct.unpack('>h', f.read(2))[0]
                # Convert to mm/3hr, handling the no-data value
                data[i, j] = np.nan if val == -9999 else val / 100.0

    return data
# return a numpy array
def convert_to_geotiff(data, geotiff_path):
    transform = from_origin(-180, 60, 0.04, 0.04)
    metadata = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': '+proj=latlong',
        'transform': transform
    }
    
    with rasterio.open(geotiff_path, 'w', **metadata) as dst:
        dst.write(data, 1)

def clip_raster_with_gpkg(raster_path, gpkg_path, clipped_raster_path):
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf.to_crs(crs='+proj=latlong')
    
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        
        out_meta.update({
            'driver': 'GTiff',
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        
        with rasterio.open(clipped_raster_path, 'w', **out_meta) as dest:
            dest.write(out_image)


def plot_clipped_data(clipped_raster_array):
    plt.figure(figsize=(12, 6))
    plt.imshow(clipped_raster_array, cmap='viridis', origin='upper')
    plt.colorbar(label='Precipitation (mm/3hr)')
    plt.title('Clipped PERSIANN-CCS Precipitation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def create_and_save_dataset(array_list, slice_data, interval, time_range):
    """
    Creates a dataset from the provided array list and saves it to a NetCDF file.

    Args:
        array_list: A list of arrays to be combined into the dataset.
        slice_data: A dictionary containing information about the slice, including
                    'start_datetime', 'end_datetime', and interval.
        time_range: 
        interval: The time interval between observations in hours.
    """
    
    dataset = xr.concat(array_list, dim=time_range.variable)
    dataset = dataset.drop_vars('spatial_ref')
    data_dict = {"precipitation": dataset}
    dataset = xr.Dataset(data_dict)
    dataset = dataset.astype(np.float32)

    slice_start = str(slice_data['start_datetime'].date()) + "HH" + format_number_with_zeros(slice_data['start_datetime'].hour, 2)
    slice_start = slice_start.replace('-', '')
    slice_end = str(slice_data['end_datetime'].date()) + "HH" + format_number_with_zeros(slice_data['end_datetime'].hour, 2)
    slice_end = slice_end.replace('-', '')
    dest_folder_name = "clipped_ds_" + str(interval) + 'h'

    os.makedirs(dest_folder_name, exist_ok=True)  # Create folder if it doesn't exist

    output_filename = slice_start + '__' + slice_end + '.nc'
    full_path = os.path.join(dest_folder_name, output_filename)
    dataset.to_netcdf(full_path)

    print(f"Clipped arrays from {slice_start} to {slice_end} saved to {full_path}")

    del dataset  # Explicitly delete dataset to free memory

def process_slice(url, nrow, ncol, shape_file, array_list):
    """
    Processes a slice of data, downloading and storing URLs.

    Args:
        array_list (list): List to store downloaded and processed data.

    Returns:
        int, int: Updated counters for total downloaded files (t_downloaded) and downloaded files within the slice (i_downloaded).
    """
        
    array_esm_temp = clip_data(read_persiann_css_online0(url, nrow, ncol), shape_file)
    array_list.append(array_esm_temp)
