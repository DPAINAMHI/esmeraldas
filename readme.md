# Introduction

## Project background
This project is part of the AdaptaClima prjoect trying to Fortify Climate Resilience in Ecuador's Coastal Cities through Empowering National Meteorological and Hydrological Services. It's a collaboration between INMAHI and UNDP.

## Project structure
This project is comprised of 4 parts, namely 4 folders including `data`, `models`, `notebooks` and `src`

- `data` folder contains all the raw data file generated during the running of all the scripts.

- `models` folder contains the prepared model after training and testing.

- `notebooks` is where all the runnable scripts are stored.

- `src` contains all the wrapped functions.

# Data Input and Processing

This project applies external satellite cloud image data due to lack of local data from the sensor stations. Therefore, before setting up the model and conduct the training process, we have to acquire all the input data from 3 different sources. This process is mainly implemented by the `data.ipynb` file.

PERSIANN-CCS is the first source of data we considered. It's a rainfall estimation dataset based on satellite infrared images of cloud-patchs operated by UC Irvine. With an update frequency of every 3 hours, we are able to downlaod the concatenated historical dataset from its data portal. The downloaded historical dataset is already saved locally in the `data` folder. In the `data.ipynb` notebook, we use the relative path to ensure it will be read correctly.

We also include WRF to expand the diversity of the input satellite cloud image data. The historical dataset is prepared and provided by INAMHI. It has a higher resolution thus it comes with a larger file size of the original historical data. For more flexible and prompt transmission, we are trying to relocate the historical dataset from local folder to the cloud. In this way, the historical dataset of WRF will be downloaded from the cloud and be temporarily stored in the memory instead of the hard disk.

Besides the external cloud image data, we still have some reliable sensors deployed already in the Esmeraldas river basin. although the number of sensors is limited, they will still enrich the perspectives of the input to increase the robustness of the model.

For the ease of data storage and trabsmission, the data of the sensors deployed in Esmeraldas city could be accessed through the `FTP` system. Using the server address, username and password written in the script, we can access the data which updates every 30 minutes with 3 different sensors (1 river level sensor and 1 precipitation sensor deployed at San Mateo and 1 river level sensor at the port).

After we finish the preparation of the forementioned data from 3 different sources, we can now unify the frequency of these datasets and merge them together for the next step. The prepared data will also be saved locally in `data/processed` to improve the efficiency of other code execution.

# Modeling and Training

The other thing we have to consider is the different arrival time form the precipitation in different pixels to the target point of the river level. Due to the different hydrological distance from different pixels, we have to include the precipitation of different pixels with different lag time. The rationale behind is that we think the further where the precipitation is, the later it will affect the river level of the downstream. 

This brings more variation to the structure of the feature matrix to be with precipitation data of different time in the same row, namely, we use different lags of the precipitation of all the pixels as the features to train the model. Practically, we divide the pixels into 10 different groups base on their distance to the river level sensor at San Mateo. The same record of the river level from the sensor is in accordance with precipitation data of 10 different timestamps.

The training of the model is rather straightforward. We will first add some new features based on the existing features. For the target indicator to predict, namely the river level at San Mateo, we will compute its rolling average of 3 periods and also include 5 lags of it.

After all the restrcuturing and expansion of the feature, we apply random forest as the primary model given its computational economy. In teh future, the determination of all hyper-parameters should be reconsidered and better apply grid-search method to fianalize it.

# Prediction Workflow










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