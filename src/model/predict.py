import numpy as np
import pandas as pd
from datetime import datetime as dt
import geopandas as gpd

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
from datetime import datetime
import xarray as xr
import rioxarray
from tqdm import tqdm
import math
from src.data.process import format_number_with_zeros, create_dataframe_from_nc



def get_adjacent_multiple_of_three_plus_one(num):
  """
  This function finds the most adjacent number that is of the form 3x + 1 
  and is no larger than the input number.

  Args:
      num: The input integer number.

  Returns:
      The most adjacent number that is of the form 3x + 1 and no larger than num.
  """
  remainder = num % 3

  # Check if the number itself is of the form 3x + 1
  if remainder == 1:
    return num
  # Otherwise, calculate the remainder when divided by 3
  # If the remainder is 0, subtract 2 to get the nearest number of the form 3x + 1
  if remainder == 0:
    if num < 2:
      return 24-2
    else:
        return num - 2
    
  # If the remainder is 2, subtract 1 to get the nearest number of the form 3x + 1
  else:
    return num - 1



  """
  Download ccs data for specified dates, processes them into a DataFrame
  """
  url = make_url(interval=3)
  if url is None:
    return None
  raw_data = read_persiann_css_online(url, 3000, 9000)
  clipped = clip_data(raw_data, shape_file)
  return clipped


def extract_datetime_from_trained_model(filename):
    date_time_str = filename.split('.')[0].split('_')[-2:]
    date_time_str = '_'.join(date_time_str)
    return datetime.datetime.strptime(date_time_str, '%Y%m%d_%H%M')


def create_2d_array(shape):
    """
    Create a 2D NumPy array of the given shape and fill it with consecutive values starting from 1.

    Args:
        shape (tuple): A tuple representing the desired shape of the 2D array (rows, columns).

    Returns:
        numpy.ndarray: A 2D NumPy array of the specified shape filled with consecutive values starting from 1.
    """
    # Calculate the total number of elements in the array
    total_elements = shape[0] * shape[1]

    # Create a 1D array with values from 1 to the total number of elements
    flat_array = np.arange(1, total_elements + 1)

    # Reshape the 1D array to the desired 2D shape
    array_2d = flat_array.reshape(shape)

    return array_2d


import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt


def modify_raster(input_file, output_file, operation, *args, **kwargs):
    """
    Modify the raster data of a GeoTIFF file, save it to a new file, and display with a legend.
    
    Args:
        input_file (str): Path to the input GeoTIFF file.
        output_file (str): Path to the output GeoTIFF file.
        operation (function): Function to perform on the raster data.
        *args, **kwargs: Additional arguments for the operation function.
    """
    with rasterio.open(input_file) as src:
        metadata = src.meta.copy()
        raster = src.read()
        
        modified_raster = operation(raster, *args, **kwargs)
        
        metadata.update({
            "driver": "GTiff", 
            "height": modified_raster.shape[1], 
            "width": modified_raster.shape[2], 
            "count": modified_raster.shape[0]
        })
        
        with rasterio.open(output_file, "w", **metadata) as dest:
            dest.write(modified_raster)
    
    # Display the modified raster with a legend
    plt.figure(figsize=(6, 4))
    
    # If the raster has multiple bands, we'll display the first band
    display_raster = modified_raster[0] if modified_raster.ndim == 3 else modified_raster
    
    # Create a colormap that excludes NaN values
    cmap = plt.cm.viridis
    cmap.set_bad('white', 1.)
    
    # Display the raster
    im = plt.imshow(display_raster, cmap=cmap)
    
    # Add colorbar as legend
    cbar = plt.colorbar(im, extend='both')
    cbar.set_label('Distance levels')
    
    # Set title
    plt.title('Esmeraldas river basin divided by distance levels to the measureing station')
    
    # Show the plot
    plt.show()


def reassign_pixel_values(raster):
    """
    Reassign pixel values to range from 1 to the overall size of the raster (including NaN values),
    based on their index occurrence order.
    
    Args:
        raster (numpy.ndarray): The raster data as a NumPy array.
        
    Returns:
        numpy.ndarray: The modified raster data with reassigned pixel values.
    """
    # Flatten the raster data into a 1D array
    flattened_raster = raster.flatten()
    
    # Get the total number of pixels in the raster (including NaN values)
    total_pixels = flattened_raster.size
    
    # Assign values from 1 to the total number of pixels based on index occurrence order
    reassigned_values = np.arange(1, total_pixels + 1)
    
    # Reshape the reassigned values to match the original raster shape
    modified_raster = reassigned_values.reshape(raster.shape)
    
    return modified_raster

from typing import List


import os
import re

def rename_gpkg_files(directory):
    """
    Rename .gpkg files in the specified directory according to the formula: new_name = 2 * old_name - 1.

    This function processes files named like '1.gpkg', '1.5.gpkg', '2.gpkg', etc.,
    and renames them to '1.gpkg', '2.gpkg', '3.gpkg', etc., respectively.

    Args:
        directory (str): The path to the directory containing the files to be renamed.

    Returns:
        None

    Side effects:
        - Renames files in the specified directory.
        - Prints messages about each renaming operation.

    Raises:
        OSError: If there are issues accessing the directory or renaming files.

    Note:
        - This function will overwrite existing files if there are naming conflicts.
        - The new file names will all be integers, losing any decimal precision from the original names.
        - Files that don't match the expected pattern (like "abc.gpkg") will be ignored.
    """
    # Compile a regular expression to match the file names
    # This pattern matches strings that start with one or more digits,
    # optionally followed by a decimal point and more digits, ending with '.gpkg'
    pattern = re.compile(r'^(\d+(?:\.\d+)?).gpkg$')
    
    try:
        # Get all .gpkg files in the directory
        gpkg_files = [f for f in os.listdir(directory) if f.endswith('.gpkg')]
        
        for old_name in gpkg_files:
            match = pattern.match(old_name)
            if match:
                # Extract the number from the file name
                num = float(match.group(1))
                
                # Calculate the new number: double the old number and subtract 1
                new_num = int(2 * num - 1)
                
                # make the number to be 2 digits to ensure correct order
                new_num = format_number_with_zeros(new_num,2)

                # Create the new file name
                new_name = f"dist_level_{new_num}.gpkg"
                
                # Construct full file paths
                old_path = os.path.join(directory, old_name)
                new_path = os.path.join(directory, new_name)
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed '{old_name}' to '{new_name}'")
            else:
                print(f"Skipped '{old_name}' as it doesn't match the expected pattern.")
    
    except OSError as e:
        print(f"An error occurred: {e}")


def map_values_to_array_indices(original_array: List[int], level_arrays: List[List[int]]) -> np.ndarray:
    """
    Maps the values of the original array to the index of each level array in the list if the value is contained
    in that specific level array.

    Args:
        original_array (List[int]): The original array containing sequential values.
        level_arrays (List[List[int]]): A list of arrays containing subsets of values from the original array.

    Returns:
        np.ndarray: A NumPy array with the same length as the original array, where each value represents the index of
        the level array in which it is present. If a value is not present in any level array, it is set to 0.
    """
    # Create a copy of the original array as a NumPy array
    mapped_array = np.array(original_array)

    # Iterate over each level array
    for i, level_array in enumerate(level_arrays):
        # Create a set of unique values in the current level array

        # Update the mapped array with the index for values present in the current level array
        mapped_array[np.isin(mapped_array, level_array)] = i + 1  # Add 1 to make the indices 1-based

    return mapped_array


def replace_with_array(raster, new_array):
    # by defualt, the shape of the raster is 3-dimensional, with the first dimension being of shape 1, which is (1,#_of_rows, #_of_cols)
    # so we compare only the 2nd and 3rd dimension with the new array
    if raster.shape[1:] != new_array.shape:
        raise ValueError("The shape of the new array must match the shape of the input raster.")
    # reshape the new_array to match the original raster's shape
    return new_array.reshape(raster.shape)


from scipy import ndimage


def replace_with_nearest_valid(array):
    """
    Replace values in a 2D array with the nearest valid value (1-10 range).
    Original NaN values are left unchanged.

    Args:
    array (numpy.ndarray): 2D input array

    Returns:
    numpy.ndarray: Modified 2D array
    """
    # Create a mask for valid values (1-10 range)
    valid_mask = (array >= 1) & (array <= 10)

    # Create a mask for values to be replaced (not NaN and not in 1-10 range)
    replace_mask = ~np.isnan(array) & ~valid_mask

    # Create a mask for original NaN values
    nan_mask = np.isnan(array)

    # Create a copy of the input array
    result = np.copy(array)

    # Replace values to be changed with a placeholder value
    placeholder = -9999  # Choose a value that doesn't occur in your data
    result[replace_mask] = placeholder

    # Use distance_transform_edt to find the nearest valid pixel
    distances, indices = ndimage.distance_transform_edt(
        ~valid_mask, 
        return_indices=True
    )

    # Replace the placeholder values with the values from the nearest valid pixel
    result[result == placeholder] = result[tuple(indices[:, replace_mask])]

    # Restore original NaN values
    result[nan_mask] = np.nan

    return result
