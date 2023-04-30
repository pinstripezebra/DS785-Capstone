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
    #try:
    if True:
        airport = pd.read_csv(df_name)
    
        airport['Time'] =  pd.to_datetime(dict(year=airport.Year, month=airport.Month, day=airport.DayofMonth))
        airport['ts'] = airport[['Time']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)

        airport.sort_values(by='Time', inplace = True)
        
        #Breakding out departure city in airlines dataframe
        airport[['city', 'OriginState']] = airport.OriginCityName.str.split(", ", expand = True)

        #Casting cities in both dataframes to uppercase
        airport['city'] = airport['city'].apply(str.upper)
        to_return = airport.iloc[:rows, :]
    #except:
    else:
        print("No Such file")
    return to_return

def clean_airline_df(df):
    
    
    '''Helper method that encodes reponse variable and drops redundant columns,
    high dimensional categorical ones, and ones that wouldnt be available in advance'''
    
    df['Delay'] = np.where(df['DepDelayMinutes'] >0,
                           1,
                           0)
    #Dropping keys
    df = df.drop(columns = ['Location-Time-KeyLocation-Time-Key','geo-location',
                           'LONGITUDELocation-Time-Key', 'DATELocation-Time-Key',
                           'Unnamed: 0Location-Time-Key', 'geo-locationLocation-Time-Key', 'LATITUDELocation-Time-Key',
                           'LATITUDELocation-Time-Key', 'Duplicate', 'Unnamed: 119','ts', 'citycity', 'citycity'])

    #Dropping information we wouldnt have until after the flight
    df = df.drop(columns = ['DepDelayMinutes','CarrierDelay', 'WeatherDelay', 'NASDelay', 
                           'SecurityDelay','ArrDelay', 'LateAircraftDelay','DepDelay',
                           'DivAirportLandings', 'DepartureDelayGroups', 'DepDel15',
                           'ArrTimeBlk', 'CancellationCode','IATA_Code_Marketing_Airline', 
                           'IATA_Code_Operating_Airline','Marketing_Airline_Network', 
                           'Operated_or_Branded_Code_Share_Partners', 'Unnamed: 0',
                           'Cancelled', 'Diverted'])
    
    df = df.drop(columns = ['WheelsOff', 'WheelsOn', 'AirTime'])
    
    #Dropping information on if the flight was diverted(wouldnt know in advace)
    df = df.drop(columns = ['DivReachedDest', 'DivActualElapsedTime', 'DivArrDelay', 
                            'DivDistance', 'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID',
                            'Div1WheelsOn', 'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff', 
                            'Div1TailNum', 'Div2Airport', 'Div2AirportID', 'Div2AirportSeqID', 
                            'Div2WheelsOn', 'Div2TotalGTime', 'Div2LongestGTime', 'Div2WheelsOff', 
                            'Div2TailNum', 'Div3Airport', 'Div3AirportID', 'Div3AirportSeqID', 
                            'Div3WheelsOn', 'Div3TotalGTime', 'Div3LongestGTime', 'Div3WheelsOff', 
                            'Div3TailNum', 'Div4Airport', 'Div4AirportID', 'Div4AirportSeqID', 
                            'Div4WheelsOn', 'Div4TotalGTime', 'Div4LongestGTime', 'Div4WheelsOff', 
                            'Div4TailNum', 'Div5Airport', 'Div5AirportID', 'Div5AirportSeqID', 
                            'Div5WheelsOn', 'Div5TotalGTime', 'Div5LongestGTime', 'Div5WheelsOff', 
                            'Div5TailNum', 'DepTime', 'DepTimeBlk', 'TaxiOut'])
    #Dropping redundant columns
    df = df.drop(columns = ['Flight_Number_Marketing_Airline', 'Originally_Scheduled_Code_Share_Airline', 
                            'DOT_ID_Originally_Scheduled_Code_Share_Airline', 'IATA_Code_Originally_Scheduled_Code_Share_Airline', 
                            'Flight_Num_Originally_Scheduled_Code_Share_Airline','DOT_ID_Operating_Airline', 'Tail_Number', 
                            'Flight_Number_Operating_Airline','DestAirportSeqID', 'DOT_ID_Marketing_Airline'])
    
    #Dropping redundant geographic information
    df = df.drop(columns = ['OriginCityName','OriginStateFips', 'OriginStateName', 'OriginWac','DestCityName',
                           'DestStateFips', 'DestStateName', 'DestWac' ])
    
    #Dropping arrival info
    df = df.drop(columns = ['ArrTime', 'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups', 'ActualElapsedTime'])
    
    return df


if __name__ == '__main__':
    
    #Loading airport data and taking subsample of x rows
    airline_df = sample_airport_df('All Aiport Data.csv', 16000000)
    
    #Loading precipitation, temperature, and city location data
    precipitation = pd.read_csv('Avg Daily Precipitation by Location.csv')
    temperature = pd.read_csv('Avg Daily Temperature by Location.csv')
    city_locations = pd.read_excel('US city data.xlsx')
    
    #Cleaning city dataframe
    city_locations['city'] = city_locations['city'].apply(str.upper)
    city_locations = city_locations[['city', 'lat', 'lng']]
    
    #Casting cities in airline to upper
    airline_df['city'] = airline_df['city'].apply(str.upper)
    print(list(airline_df.columns))
    #Merging dataframes and adding lat/long key to lookup weather
    airline_df = airline_df.join(city_locations, lsuffix = 'city',rsuffix = 'city')
    airline_df['Location-Time-Key'] = list(zip(airline_df['lat'].round(0),  
                                               airline_df['lng'].round(0), 
                                               airline_df['FlightDate']))
    temperature['Location-Time-Key'] = list(zip(temperature['LATITUDE'].round(0),  
                                                temperature['LONGITUDE'].round(0), 
                                                temperature['DATE']))
    precipitation['Location-Time-Key'] = list(zip(precipitation['LATITUDE'].round(0),  
                                                  precipitation['LONGITUDE'].round(0), 
                                                  precipitation['DATE']))
    wind['Location-Time-Key'] = list(zip(wind['LATITUDE'].round(0),  
                                                  wind['LONGITUDE'].round(0), 
                                                  wind['DATE']))
    
    #Now joining airline dataset with temperature/precipitation/wind
    airline_df = airline_df.join(temperature, lsuffix = 'Location-Time-Key',
                                 rsuffix = 'Location-Time-Key')
    airline_df = airline_df.join(precipitation, lsuffix = 'Location-Time-Key',
                                 rsuffix = 'Location-Time-Key')
    airline_df = airline_df.join(wind, lsuffix = 'Location-Time-Key',
                                 rsuffix = 'Location-Time-Key')
    airline_df = clean_airline_df(airline_df)
    
    #Now writing to combined csv file for use in Time Series Forecasting script
    airline_df.to_csv('large sample, 16M.csv')
