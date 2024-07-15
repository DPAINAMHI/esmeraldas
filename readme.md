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

After all the restrcuturing and expansion of the feature, we apply random forest as the primary model given its computational economy. In the future, the determination of all hyper-parameters should be reconsidered and better apply grid-search method to fianalize it.

# Prediction Workflow

The prediction is simply to input prepared features of the future into the trained model to produce the output. However, the obstacal is with the prepration of the dynamically updating features.

Unlike the static historical data, the updating of the features is dynamic. We have to access them every 3 hours with 3 different sources.Using the three functions to update the features of the CCS, WRF and sensors at Esmeraldas seperaetly, we run them every 3 hours to get the most recent update of the features to input.

The dynamic updating timeframe of different sources of data is rather tricky. First the WRF data will update 1 day ahead. Since it's a prediction of 3 days, we will use the most recent prediction of the precipitation data of all the pixels within the Esmeraldas river basin. And the most recent lag data (i.e. lag 1) from the sensors deployed in Esmeraldas stations will update within 30 minutes after the actual time, that is to say the lag 1 data of the sensor will be available 2.5 hours ahead of the time to predict. Lastly, the update time of the CCS data from its data portal is the most unstable. But most likely it will update 1.5 hours ahead of the target timestamp to predict.

After running the trunk of code to fetch different dynamic features every 3 hours, we consolidate those features (be aware that some of the features might be its own lag) into a feature array and pass it to the pre-trained model, in which way we generate our result of the prediction.

# Future To-dos
## Underlying progress

- Replace the static hyper-parameter of the model training with grid-search results
- Complete the storage transfering of the wrf historical static data from the local to the cldou site
- Finish the prediction workflow
    - Find the reliable portal of fetching the WRF data stably and dynamically
    - Enable the auto-downloading and cleansing of the WRF data
- Figure out the unified updating timeframe of the dynamic features
- Documentation and transmission of the prediction result to the existing information system

## Potential improvement

- Further analysis of the prediction result and the corresponding push notifications of alerts once the prediction exceeds the threshold
- Inclusion of more types of input data to consolidate the features