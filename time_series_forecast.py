import numpy as np
import pandas as pd
#For preprocessing and model training
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#Visualizing
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


def create_dataset(dataset, look_back=1):
    '''Helper function that takes a subset of input dataset based on time, and returns subsetted data.
    Provides the data to enable a time series model to predict future values based on previous 'look_back' time period
    '''
    
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_airport_delay_models(grouped_df, look_back, epochs, start_date):
    
    '''Inputs
    ------------
    (1) grouped_df: Dataframe containing 3 columns: time, airport_id, delay chance for day
    (2) look_back: Integer specifying period of time to train over
    Returns
    --------
    models: Dictionary containing models trained for each unique departing airport id'''
    
    models = {}
    airports = grouped_df['OriginAirportID'].unique()
    for airport in airports:
        try:
            df = grouped_df[grouped_df['OriginAirportID'] == airport]
            tf.random.set_seed(7)
            #Extracting delay data for airport as numpy array
            dataset = df.percent_delay.to_numpy().reshape(-1, 1)
            dataset = dataset.astype('float32')

            #Normalizing and transforming data for airport
            scaler = MinMaxScaler()
            scaler.fit(dataset)
            dataset = scaler.transform(dataset)

            #Splitting into test and train sets sequentially
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            #Reshaping numpy array to train model based upon x time steps
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            # create and fit the LSTM network
            model = Sequential()
            model.add(layers.LSTM(4, input_shape=(1, look_back)))
            model.add(layers.Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=1)

            # make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            #Measuring RMSE for train and test sets
            trainScore = np.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
            testScore = np.sqrt(mean_squared_error(testY, testPredict[:,0]))


            prediction_list = list(trainPredict[:,0]) + list(testPredict[:,0])
            airport_list = [airport] * len(prediction_list)
            out = pd.DataFrame(list(zip(dates, airport_list, prediction_list)),
                              columns = ['date', 'airport', 'prediction_local'])

            #Adding predictions/model to output dictionary
            models[airport] = {'model': model,
                              'trainRMSE': trainScore,
                              'testRMSE': testScore,
                              'prediction_df': out}
        except:
            print(str(airport) +" had error in Script")
    return models

'''PART1: Performing time series forecasting and adding prediction to dataframe'''

#Reading in dataset and sorting by time
df = pd.read_csv('large sample, 16M.csv',
                low_memory=False)
df.sort_values(by='Time', inplace = True)

#Defining list of departing airports
departing_airports = df['OriginAirportID'].unique()
df_sample = df
df_sample.fillna(0)

#Subsetting for time prediction
df_sample_ts = df_sample[['Time','OriginAirportID','Delay']]
df_sample_ts['Time'] = pd.to_datetime(df_sample_ts['Time'])
df_sample_ts = df_sample_ts.sort_values(by=['Time', 'OriginAirportID'])

#Aggregating to day/airport level
grouped = df_sample_ts.groupby(by=['Time','OriginAirportID']).agg(
    total_flights=('Delay', np.any),
    percent_delay=('Delay', np.mean)).reset_index()
for_train = grouped.drop(columns = ['total_flights'])

#Aggregating to day level
grouped_total = df_sample_ts.groupby(by=['Time']).agg(
    total_flights=('Delay', np.any),
    percent_delay=('Delay', np.mean)).reset_index()
for_train_total = grouped_total.drop(columns = ['total_flights'])

#Splitting into test and train set
#Splitting by date since its timeseries
test_cutoff_date = for_train_total['Time'].max() - timedelta(days=3)
val_cutoff_date = test_cutoff_date - timedelta(days=6)

df_test = for_train_total[for_train_total['Time'] > test_cutoff_date]
df_val = for_train_total[(for_train_total['Time'] > val_cutoff_date) & (for_train_total['Time'] <= test_cutoff_date)]
df_train = for_train_total[for_train_total['Time'] <= val_cutoff_date]

#check out the datasets
print('Test dates: {} to {}'.format(df_test['Time'].min(), df_test['Time'].max()))
print('Validation dates: {} to {}'.format(df_val['Time'].min(), df_val['Time'].max()))
print('Train dates: {} to {}'.format(df_train['Time'].min(), df_train['Time'].max()))

#Plotting line graph of delay over time
for_train_total.head()
sns.lineplot(x="Time", y="percent_delay", data=for_train_total)
plt.title('Delay Percentage by Time - United States Domestic')
plt.xticks(rotation = 90)
plt.show()

#Training LSTM on dataset
tf.random.set_seed(7)
dataset = for_train_total.percent_delay.to_numpy().reshape(-1, 1)
dataset = dataset.astype('float32')

#Normalizing Dataset
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

#Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#Reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#Reshaping input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#Creating and fit the LSTM network
model = Sequential()
model.add(layers.LSTM(4, input_shape=(1, look_back)))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

##Making Predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#Assessing accuracy of predictions
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#Shifting data and plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title('Delay Prediction vs Actual - Global')
plt.show()

dates = for_train['Time'][1:]
models = train_airport_delay_models(for_train, 1, 50, dates)

#Assembling predictions into single dataframe
total_predictions = []
count = 0
for i in list(models.keys()):
    if count == 0:
        total_predictions = models[i]['prediction_df']
    else:
        total_predictions = pd.concat([total_predictions, models[i]['prediction_df']])
    count += 1
    
#Adding airport/time key for use in joining prediction back to main dataframe
total_predictions['key'] = total_predictions['date'].astype(str) + total_predictions['airport'].astype(str)

#Adding predictions to dataframe
df['key'] = df['Time'].astype(str)+ df['OriginAirportID'].astype(str)
out_df = pd.merge(df, total_predictions, how = 'left', on = 'key')
out_df.head(10)

'''PART2:Performing day/airport level capacity calculations and adding results to dataframe'''
#Aggregating planes by day and departure airport
airport_utilization = out_df[['Time','OriginAirportID']].groupby(by = ['Time','OriginAirportID'] ).size().reset_index(name='counts')
airport_utilization = airport_utilization.reset_index().drop(columns = ['index'])
header_row = ['Time','OriginAirportID', 'Plane_Count']
util2 = pd.DataFrame(airport_utilization.values[1:], columns = header_row)

#Calculating Max airplanes per airport
max_planes = util2[['OriginAirportID', 'Plane_Count']].groupby(by = ['OriginAirportID']).agg({'Plane_Count':['max']}).reset_index()
header_row_2 = ['OriginAirportID', 'Max_Plane_Count']
max_planes_2 = pd.DataFrame(max_planes.values[1:], columns = header_row_2)

#Examining data
max_planes_2.head(10)

#Converting keys to strings
max_planes_2['OriginAirportID'] = max_planes_2['OriginAirportID'].astype(str)
util2['OriginAirportID'] = util2['OriginAirportID'].astype(str)

#Joining two dataframes together to add max_planes variable to daily capacity dataset
joined_planes = pd.merge(util2, max_planes_2, how = 'left', on = 'OriginAirportID')
joined_planes['Percentage_Capacity'] = joined_planes['Plane_Count']/joined_planes['Max_Plane_Count']

# Now Joining back to the main dataset
out_df['Capacity_key'] = out_df['Time'].astype(str) + out_df['OriginAirportID'].astype(str)
joined_planes['Capacity_key'] = joined_planes['Time'].astype(str) + joined_planes['OriginAirportID'].astype(str)
out_df = pd.merge(out_df, joined_planes, how = 'inner', on = 'Capacity_key')

#Now dropping key
out_df = out_df.drop(columns = ['Capacity_key'])

#Finally Outputting dataframe for use in modeling script
out_df.to_csv('large sample, df 16M, with forecast.csv')
