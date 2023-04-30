# DS785-Capstone
Repository containing code for a Masters Thesis analyzing causes of airline delays

# Code Description:
1. Data pre-processing: 
* Joins and cleans airline and weather datasets
* Drops unnescessary and redundant variables
* Outputs combined CSV file for use in Time Series Forecast
2. time_series_forecast: 
* Uses LSTM model to forecast % of flights delayed from each airplane 24 hours in advance, adds this forecast to the dataset
* Calculates airport level capacity metrics and adds these to dataset
* Outputs CSV file for use in Airline Modeling Script
3. Airline_modeling: 
* Performs encoding on categorical variables
* Trains models in 2-hour and 24-hour prediction scenarios
* Assesses feature importance of each model in each scenario
* Visualizes model training and feature importance for use in presentation
