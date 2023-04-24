# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:18:10 2023

@author: seelc
"""
import pandas as pd
import numpy as np


def sample_airport_df(df_name, rows):
    
    '''Takes input csv filename, sorts by time, and returns sample as DataFrame
    Inputs
        df_name: name of file
        rows: Number of observations to include in output
    -----------------------------------------------
    Returns
        to_return: Dataframe
    
    '''
    
    to_return = ""
    try:
        airport = pd.read_csv(df_name)
    
        airport['Time'] =  pd.to_datetime(dict(year=airport.Year, month=airport.Month, day=airport.DayofMonth))
        airport['ts'] = airport[['Time']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)

        airport.sort_values(by='Time', inplace = True)
        
        #Breakding out departure city in airlines dataframe
        airport[['city', 'OriginState']] = airport.OriginCityName.str.split(", ", expand = True)

        #Casting cities in both dataframes to uppercase
        airport['city'] = airport['city'].apply(str.upper)
        to_return = airport.head(rows)
    except:
        print("No Such file")
    return to_return

def clean_airline_df(df):
    
    
    '''Helper method that encodes reponse variable and drops redundant columns,
    high dimensional categorical ones, and ones that wouldnt be available in advance'''
    
    df['Delay'] = np.where(df['DepDelayMinutes'] >0,
                           1,
                           0)
    df = df.drop(columns = ['citycity', 'citycity.1','DATELocation-Time-Key.1',
                            'geo-locationLocation-Time-Key.1', 'Location-Time-Key',
                            'Location-Time-KeyLocation-Time-Key',
                            'geo-locationLocation-Time-Key', 'DATELocation-Time-Key',
                            'Unnamed: 0Location-Time-Key.1', 'LATITUDELocation-Time-Key',
                            'Cancelled', 'Diverted','LATITUDELocation-Time-Key.1', 
                            'LONGITUDELocation-Time-Key.1','LONGITUDELocation-Time-Key',
                            'Unnamed: 0.1','Location-Time-KeyLocation-Time-Key.1'])
    df = df.drop(columns = ['DepDelayMinutes','CarrierDelay', 'WeatherDelay', 'NASDelay', 
                           'SecurityDelay','ArrDelay', 'LateAircraftDelay','DepDelay',
                           'DivAirportLandings', 'DepartureDelayGroups', 'DepDel15',
                           'ArrTimeBlk', 'CancellationCode','IATA_Code_Marketing_Airline', 
                           'IATA_Code_Operating_Airline','Marketing_Airline_Network', 
                           'Operated_or_Branded_Code_Share_Partners'])
    
    return df


if __name__ == 'main':
    
    #Loading airport data
    airline_df = sample_airport_df('All Aiport Data.csv', 4000000)
    
    #Loading precipitation, temperature, and city location data
    precipitation = pd.read_csv('Avg Daily Precipitation by Location.csv')
    temperature = pd.read_csv('Avg Daily Temperature by Location.csv')
    city_locations = pd.read_excel('US city data.xlsx')
    
    #Cleaning city dataframe
    city_locations['city'] = city_locations['city'].apply(str.upper)
    city_locations = city_locations[['city', 'lat', 'lng']]
    
    #Merging dataframes and adding lat/long key to lookup weather
    airline = airline_df.join(city_locations, lsuffix = 'city',rsuffix = 'city')
    airline_df['Location-Time-Key'] = list(zip(airline_df['lat'].round(0),  
                                               airline_df['lng'].round(0), 
                                               airline_df['FlightDate']))
    temperature['Location-Time-Key'] = list(zip(temperature['LATITUDE'].round(0),  
                                                temperature['LONGITUDE'].round(0), 
                                                temperature['DATE']))
    precipitation['Location-Time-Key'] = list(zip(precipitation['LATITUDE'].round(0),  
                                                  precipitation['LONGITUDE'].round(0), 
                                                  precipitation['DATE']))
    
    #Now joining airline dataset with temperature/precipitation
    airline_df = airline_df.join(temperature, lsuffix = 'Location-Time-Key',
                                 rsuffix = 'Location-Time-Key')
    airline_df = airline_df.join(precipitation, lsuffix = 'Location-Time-Key',
                                 rsuffix = 'Location-Time-Key')
    
    #Now cleaning joined airline df
    airline_df = clean_airline_df(airline_df)
    
    #Writing to output file
    airline_df.to_csv
    