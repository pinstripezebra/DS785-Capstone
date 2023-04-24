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
#Reading in dataset
df = pd.read_csv('C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Masters Acquisition\\Capstone\\Data for Modeling\\medium_Sample_Model, 4M.csv',
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

# normalize the dataset
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
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
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#Assessing accuracy of predictions
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Shifting data and plotting
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

total_predictions['key'] = str(total_predictions['date']) + str(total_predictions['airport'])

#Adding predictions to dataframe
df['key'] = str(df['Time'])+ str(df['OriginAirportID']) 
out_df = df.join(total_predictions, how='left', lsuffix='key', rsuffix='key')

#Outputting dataframe
out_df.to_csv('Medium Sample, df 4M.csv')
