# Introduction

This project is comprised of a main jupyter notebook `CCS_load_data.ipynb` and supporting functions composed in `funct.py` and `download_func.py`. 

# Data Preparation
## CCS Data
PERSIANN-Cloud Classification System (PERSIANN-CCS) is a real-time global high resolution (0.04° x 0.04° or 4km x 4km;) satellite precipitation product developed by the Center for Hydrometeorology and Remote Sensing (CHRS) at the University of California, Irvine (UCI)

There are 2 parts of the CCS data:
- Static historical data: Download mannually from the UCI data portal with a pre-clipped form in the `.nc` format
- Most recent data to be updated: To track and update the most recent updated data using a python script automatically

With the combination of these 2 sources of data, we formulate the dataset preparation.

## Esmeraldas Sensor Data
Except for the satellite CCS data, we also adopt local data from the sensors deployed in the Esmeraldas. The sensored data comes from two stations and are updated through the FTP system by CAE.
The station-sensor structure is as follows:
- Autoridad Portuaria, Estación A - Muelle 1 (`m1`), Code: 945700
    - Sensor de nivel de agua, 78652, Hydrometric level, `hidro_level_m1`
- San Mateo (`sm`), Estación B, Code: 945800
    - Sensor de nivel de agua, 78694, Hydrometric level, `hidro_level_sm`
    - Sensor de precipitación, 78656, precipitación acumulada, `precip_acumu_sm`

AS shown above, different variable name was assigned to the data coming from each sensor. Those variable names are also in accordance with the `colnames` of the final output `pd.dataframe`.
Right now, we have the data from the FTP system ranging from `20230906` to `20240308`

## WRF Data
TBD

# Input
## Read the historical CCS data
In the main jupyter notebook, to read the historical data, first we will have to specify the frequency of the CCS data we used. By default, we will use the 3-hour frequency, namely `freq = 3`.
Next, we will have to specify the relative path of the folder where the data is located. Note that we will get the current directory of the jupyter notebook and return to the root folder of this project where also the relative path of the data is based on.
Thus if you wanna use the data with a different frequncy, except for changing the input of the frequency, you will also have to download the historical data in advance and change the relative path accordingly if necessary.
## Read the historical Esmeraldas sensor data
The input of the code for reading the historical Esmeraldas sensor data is just the start and end date specifying the time range of the data.

# Output
## Output of the CCS clipped data
The downloaded CCS data is pre-clipped, namely consisting of only the pixels of the esmeraldas river basin.
According to the prescribed information associated with the downloaded data, the shape of the Esmeraldas river basin is $53\times52$. As part of the processing of the data, we flatted the data to be 1-dimensional and drop all the null pixels outside the actual basin shape. As the result, the net output shape of the CCS clipped data is $1\times 1098$.
## Output of the Esmeldas sensor data
Since there are only 3 available sensors deployed in the Esmeraldas area, the output shape is simply three columns with each indicating the time series data from 3 sensors with the exception from the date 2023/10/4 to 2023/10/21 when there is only one sensor working properly, which is the `hidro_level_m1`.
And as of 2024/3/26, the data from 2023/09/01 to 2023/09/05 is still missing.
Note that the orginal frequency of these 3 time series is 1 minute or 5 minutes. To be compatible with main CCS data, we aggregate/pivot the time series to change its frequency to be 3 hours. For the precipitaion data, the aggregation method is `diff`, namely get the difference between the observation of the last and first timestamps in the unit time. For the water level data, the aggregation method is averaging.
The names of the aggregated columns are `hidro_level_m1`， `precip_acumu_sm`, `hidro_level_sm` accordingly indicating the average water level of Estación A - Muelle 1, precipitación acumulada of Estación B San Mateo and average water level of Estación B San Mateo.
## Final merged data
The output of the above acquiry of historical data is simply two sperate dataframes. To formulate the general dataset for further forecasting, we merge these two timestamp-indexed based on the timestamp of each row. Hence the output overall dataframe consists of $1098+3=1102$ columns.
Due to the different available time range of these data sources and the outter way of merging, the merged dataset is quite staggered, i.e. the number of available rows containing all non-null data is limited.