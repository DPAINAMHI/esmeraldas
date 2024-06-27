import pandas as pd
import gzip
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import requests
import math
from ftplib import FTP
import os
from datetime import datetime, timedelta

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


def download_esm_station_data(local_base_folder, ftp_server, username, password):
    # FTP server details
    ftp = FTP(ftp_server)
    ftp.login(username, password)

    # Change to the desired directory
    ftp.cwd("TEST_ESTACIONES_AUTOMATICAS/H5033/D1/F10")

    # Find the latest date of existing files in the local folder
    latest_local_date = None
    for root, dirs, files in os.walk(local_base_folder):
        for file in files:
            if file.endswith(".csv"):
                file_date = datetime.strptime(file[:8], "%Y%m%d").date()
                if latest_local_date is None or file_date > latest_local_date:
                    latest_local_date = file_date

    # List all the files and folders in the current directory
    files_and_folders = ftp.nlst()

    # Filter out the folders starting with "yd"
    yd_folders = [folder for folder in files_and_folders if folder.startswith("yd")]

    for yd_folder in yd_folders:
        # Extract the year from the "yd" folder name
        yd_year = int(yd_folder[2:])
        
        # Check if the "yd" folder contains CSV files later than the latest local date
        if latest_local_date is None or yd_year >= latest_local_date.year:
            # Change to the "yd" folder
            ftp.cwd(yd_folder)
            files_and_folders = ftp.nlst()
            
            # Filter out the folders starting with "md"
            md_folders = [folder for folder in files_and_folders if folder.startswith("md")]
            
            for md_folder in md_folders:
                # Extract the year and month from the "md" folder name
                md_year = int(md_folder[2:6])
                md_month = int(md_folder[6:])
                
                # Check if the "md" folder contains CSV files later than the latest local date
                if latest_local_date is None or (md_year, md_month) >= (latest_local_date.year, latest_local_date.month):
                    # Change to the "md" folder
                    ftp.cwd(md_folder)
                    
                    # List all the CSV files in the current folder
                    csv_files = [file for file in ftp.nlst() if file.endswith(".csv")]
                    
                    for csv_file in csv_files:
                        # Extract the date from the CSV file name
                        file_date = datetime.strptime(csv_file[:8], "%Y%m%d").date()
                        
                        # Set the specific time to be 23:55 of the file date
                        specific_datetime = datetime.combine(file_date, datetime.strptime("23:55", "%H:%M").time())
                        
                        # Add one day to the specific time
                        specific_datetime += timedelta(days=1)
                        
                        # Get the last modified time of the CSV file
                        last_modified_time = ftp.voidcmd(f"MDTM {csv_file}")[4:].strip()
                        last_modified_datetime = datetime.strptime(last_modified_time, "%Y%m%d%H%M%S")
                        
                        # Check if the last modified time is later than the specific time
                        if last_modified_datetime >= specific_datetime:
                            # Create the year and month folders if they don't exist
                            local_year_folder = os.path.join(local_base_folder, yd_folder)
                            local_month_folder = os.path.join(local_year_folder, md_folder)
                            os.makedirs(local_month_folder, exist_ok=True)
                            
                            # Check if the corresponding CSV file already exists
                            local_file_path = os.path.join(local_month_folder, csv_file)
                            if not os.path.exists(local_file_path):
                                # Download the CSV file to the local month folder
                                with open(local_file_path, "wb") as file:
                                    # Use FTP's RETR command to download the file
                                    ftp.retrbinary(f"RETR {csv_file}", file.write)
                                print(f"Downloaded {csv_file} to {local_month_folder}")
                        else:
                            print(f"Last modified time of {csv_file}: {last_modified_datetime} is not later than {specific_datetime}. Collection of sensor data unfinished. Skipping download.")
                    
                    ftp.cwd("..")  # Go back to the parent directory
            
            ftp.cwd("..")  # Go back to the parent directory
    ftp.quit()
    return file_date



import requests

def download_file_from_onedrive(url, output_path):
    # Send a GET request to the OneDrive URL
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the output file in binary write mode
        with open(output_path, 'wb') as file:
            # Write the content to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully to {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")