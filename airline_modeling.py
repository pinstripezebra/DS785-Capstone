# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
#Pre-processing packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#For preprocessing and model training
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from sklearn import preprocessing
#Visualizing
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import seaborn as sns
import tensorflow_addons as tfa
#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import re

#Model Explainability and scoring
import shap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

        
def drop_cols(df):
        
    '''Helper Function to drop columns from dataframe
    INPUT:
        df: Dataframe to drop columns from
    --------------------------------------
    OUTPUT:
        df: Dataframe with target columns dropped'''
        
    df = df.drop(columns = ['WheelsOff', 'WheelsOn', 'AirTime'])
    df = df.drop(columns = ['keykey', 'Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0Location-Time-Key',
                       'ts', 'Duplicate','ActualElapsedTime', 'ArrDel15', 'Unnamed: 0.2', 'keykey.1', 'DOT_ID_Marketing_Airline',
                       'Flight_Number_Marketing_Airline', 'Flight_Number_Operating_Airline', 'DepTime'])
    return df
        
def encode_dummies(df_sample):
        
    '''Performs one hot encoding on input dataframe to encode n level categorical variables
    as n binary ones. Returns encoded dataframe'''

    #One Hot Encoding State name
    state_encode = pd.get_dummies(df_sample.OriginStateName, prefix='State')
        
    #Adding encoded variables to sample df
    df_sample = pd.concat([df_sample, state_encode], axis =1)
    y = df_sample['Delay']
    x = df_sample.drop(columns = ['Delay'])
    
    return x, y
    
def normalize_df( numeric_df):
        
    '''Takes numeric_dataframe as input, calculates the skewness of each variable, and normalizes any
    variables with a skewness score of >0.5, returns normalized dataframe''' 

    col_list = []
    for i in numeric_df.columns:
        if i.find('State')!=0:
            col_list.append(i)
                
    for_visual = numeric_df.loc[:, numeric_df.columns.isin(col_list)].sample(1000)
        
    skewness = for_visual.skew(axis = 0)
    kurtosis = for_visual.kurt(axis = 0)
    initial_assessment = pd.DataFrame(np.array([for_visual.columns, skewness, kurtosis]).T,
                                         columns = ['Variable', 'Skewness', 'Kurtosis'])
    initial_assessment['Skew Clasification'] = np.where((abs(initial_assessment['Skewness']) < 0.5) , 'Symmetrical', 'Skewed')
    initial_assessment.head(n=20)
    attributes = initial_assessment[initial_assessment['Skew Clasification'] == 'Skewed']['Variable']
    numeric_df[attributes] = preprocessing.normalize(numeric_df[attributes])
        
    return numeric_df

def select_best_model_f1(histories):
    
    '''Takes input list of models and returns model with best F1 Score'''
    
    index = 0
    best_accuracy = 0
    keys = list(histories.keys())
    for model in range(len(histories.keys())):
        accuracy = round(histories[keys[model]]['Results']['val_f1_score'].max(),4)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            index = model
        
    return index

def visualize_f1(histories):
    
    '''Takes dataframe containing keras training results, returns grid 
        displaying model accuracy over training epoch'''
    
    graph_count = len(histories)
    rows = 2
    columns = int(math.ceil(graph_count/rows))
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=rows, ncols=columns)
    keys = list(histories.keys())
    for model in range(len(histories.keys())):
        ax = fig.add_subplot(gs[int(math.floor(model/columns)), model%columns])
        ax.plot(histories[model]['Results'].index, histories[model]['Results']['f1_score'], label="Train f1")
        ax.plot(histories[model]['Results'].index, histories[model]['Results']['val_f1_score'], label="Val f1")
        plt.legend(loc="upper left")
        plt.grid(True)
        
        #Returning accuracy and formatting
        accuracy = round(histories[keys[model]]['Results']['val_f1_score'].max(),4)
        plt.title("Model {graph} f1 score {accuracy}".format(graph = model,
                                                 accuracy = accuracy))
        
    plt.tight_layout()
    plt.show()      

def confusion_matrices(cf_matrix):
    
    '''Takes dictionary of confusion matrice's and graphs then in single figure'''
    fig, axn = plt.subplots(1,3, sharex=True, sharey=True,figsize=(12,4))

    for i, ax in enumerate(axn.flat):
        k = list(cf_matrix)[i]
        sns.heatmap(cf_matrix[k], ax=ax,cbar=i==4,annot=True, cmap="crest")
        ax.set_title(k,fontsize=8)
    fig.suptitle('Confusion Matrix - Model Comparison, 2 Hour Window')
    fig.supxlabel('Predicted Value')
    fig.supylabel('True Value')
    plt.tight_layout()
    plt.show()
    
def fit_regressor(X_train, y_train, X_test, y_test):
        
    #Fitting Model
    reg_model = LogisticRegression(random_state=42, penalty = 'l2', max_iter = 1000, class_weight = 'balanced').fit(X_train, y_train)
    #Making predictions
    reg_prediction = reg_model.predict(X_test)
    #Measuring accuracy
    reg_accuracy = accuracy_score(y_test, reg_prediction)
    
    return reg_model, reg_accuracy
        
def fit_tree(X_train, y_train, X_test, y_test):
    #Defining parameters for hyperparameter search
    depth = [2,4,6]
    number_estimators = [64, 128, 256]
    param_grid = dict(max_depth = depth, n_estimators = number_estimators)
    #Building the gridsearch
    rf = RandomForestClassifier(n_estimators = number_estimators, max_depth = depth)
    grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3)
    grid_results = grid.fit(X_train, y_train)
        
    #Extracting best Random Forest model
    best_rf = grid_results.best_estimator_
    tree_prediction = best_rf.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_prediction)
    
    return best_rf, tree_accuracy


def create_baseline_small(input_df):
    input_features = input_df.shape[1]
    model = Sequential()
    model.add(Dense(input_features, input_shape=(input_features,), activation='relu'))
    model.add(Dropout(.2, input_shape=(input_features,))) #Dropout layer to avoid overfitting
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics= [tfa.metrics.F1Score(num_classes = 2, threshold = 0.5, average="micro")])
    return model

def create_baseline_medium(input_df):
    input_features = input_df.shape[1]
    model = Sequential()
    model.add(Dense(input_features, input_shape=(input_features,), activation='relu'))
    model.add(Dropout(.2, input_shape=(input_features,))) #Dropout layer to avoid overfitting
    model.add(Dense(input_features, input_shape=(input_features*2,), activation='relu')) #Second layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=[tfa.metrics.F1Score(num_classes = 2, threshold = 0.5, average="micro")])
    return model

def create_baseline_large(input_df):
    input_features = input_df.shape[1]
    model = Sequential()
    model.add(Dense(input_features, input_shape=(input_features,), activation='relu'))
    model.add(Dropout(.2, input_shape=(input_features,))) #Dropout layer to avoid overfitting
    model.add(Dense(input_features, input_shape=(input_features*2,), activation='relu')) #Second layer
    model.add(Dense(input_features, input_shape=(input_features,), activation='relu'))  #third layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=[tfa.metrics.F1Score(num_classes = 2, threshold = 0.5, average="micro")])
    return model

def sklearn_feature_importance(best_rf, reg_model):
    
    '''Takes random forest and regression models as inputs and graphs feature importance'''
    
    tree_1_features = best_rf.feature_importances_

    forest_importances = pd.Series(tree_1_features,  index=X_train.columns)
    forest_importances = forest_importances.to_frame(name = 'feature_importance')
    forest_importances = forest_importances.sort_values(by = 'feature_importance', ascending = False)[0:10]
    
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True,figsize=(12,4))
    

    sns.barplot(data=forest_importances, x=forest_importances.index, y="feature_importance", ax = ax[0])
    ax[0].set_title("Feature importances using MDI - RF Classifier")
    ax[0].set_ylabel("Mean decrease in impurity")
    ax[0].set_xticklabels(forest_importances.index, rotation=90, ha='right')
    
    #Creating dataframe
    reg_importances = pd.Series(tree_1_features,  index=X_train.columns)
    reg_importances = reg_importances.to_frame(name = 'feature_importance')
    reg_importances = reg_importances.sort_values(by = 'feature_importance', ascending = False)[0:10]
    
    sns.barplot(data=forest_importances, x=forest_importances.index, y="feature_importance", ax = ax[1])
    ax[1].set_title("Feature importances using MDI - Logistic Regressor")
    ax[1].set_ylabel("Variable Coefficient")
    ax[1].set_xticklabels(reg_importances.index, rotation=90, ha='right')
    
    fig.suptitle('Predictor Importance, 2 Hour Window')
    fig.tight_layout()
    plt.show()
    
def neural_network_feature_importance(best_model, X_test):
    
    '''Takes model and X_test dataframe as inputs and displays shapley plots for feature importance'''
    
    # Fits the explainer
    explainer = shap.Explainer(best_model.predict, X_test.iloc[50:150,:])
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_test.iloc[50:100,:])
    
    # plotting values
    shap.plots.bar(shap_values)
    
    #Violin plot
    shap.summary_plot(shap_values)
    
if __name__ == '__main__':
    
    #Load dataframe, sort by time, remove columns
    df = pd.read_csv('large sample, df 16M, with forecast.csv')
    df.sort_values(by='Time', inplace = True)
    df = drop_cols(df)
    
    #Encoding Categorical Variables
    x, y = encode_dummies(df)
    
    #Selecting numberic variables and normalizing skew ones
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']
    numeric_df = x.select_dtypes(include=numerics)
    numeric_df = normalize_df(numeric_df)
    
    #Spltting into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(
        numeric_df, y, test_size=0.33, random_state=42)
    
    '''Scenario 1: 2 hour prior to departure, have previous flight information'''
    #Fitting Regressor and tree models
    regressor_model, regressor_accuracy = fit_regressor(X_train, y_train, X_test, y_test)
    tree_model, tree_accuracy = fit_tree(X_train, y_train, X_test, y_test)
    
    #Graphing feature importance for random forest and logistic regression
    sklearn_feature_importance(tree_model, regressor_model)
    
    #Training neural network on subset of data to identify best later/node count
    #Performing preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train, X_test)
    cols = X_train.columns
    X_train = pd.DataFrame(scaler.transform(X_train), columns = cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = cols)


    x_train_small = X_train.iloc[:100000, :]
    y_train_small = y_train.iloc[:100000]
    x_test_small = X_test.iloc[:100000, :]
    y_test_small = y_test.iloc[:100000]

    #Creating models for Training
    models = [create_baseline_small(x_train_small), create_baseline_medium(x_train_small), create_baseline_medium(x_train_small)]
    names = ['Small', 'Medium', 'Large']
    accuracy = []
    histories = {}

    #Training models
    for i in range(len(models)):
        history = models[i].fit(x_train_small, y_train_small, epochs = 25,verbose= 1,
                            validation_data = (x_test_small, y_test_small))
        accuracy.append(models[i].evaluate(x_test_small, y_test_small))

        histories[i] = {'Results' :pd.DataFrame(history.history),
                            'Model' : history,
                            'End Score': history.history['val_f1_score'][-1]}
        
    #Viualizing training histories
    visualize_f1(histories)
        
    #Returning best model and training on entire dataset
    best_model_index = select_best_model_f1(histories)
    best_model = models[best_model_index]
    best_model.fit(X_train, y_train, epochs = 25, verbose = 1,
                            validation_data = (X_test, y_test))
    #Returning model summary   
    best_model.summary()
        
    #Making predictions with all models and putting into dictionary
    reg_predictions = np.round(regressor_model.predict(X_test),0)
    tree_predictions = np.round(tree_model.predict(X_test),0)
    nn_predictions = np.round(best_model.predict(X_test),0).astype(int)
    cf_matrix = {'Logistic Classifier': confusion_matrix(y_test, reg_predictions, normalize = 'all') ,
                "RandomForest Clasifier": confusion_matrix(y_test, tree_predictions,normalize = 'all'),
                'Neural Network': confusion_matrix(y_test, nn_predictions, normalize = 'all')}
    
    #Generating confusion matrice's for all models predictions
    confusion_matrices(cf_matrix)
    
    #Visualizing feature importance for random forest and logistic regression models
    sklearn_feature_importance(tree_model, regressor_model)
    neural_network_feature_importance(nn_model, X_test)
    
    #Using shap library to visualize variable importance for neural networks
    explainer = shap.KernelExplainer(best_model, X_train.iloc[:50,:])
    shap_values = explainer.shap_values(X_train.iloc[20,:], nsamples=500)
    shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[20,:])

    shap_values50 = explainer.shap_values(X_train.iloc[50:100,:], nsamples=500)
    shap.force_plot(explainer.expected_value, shap_values50[0], X_train.iloc[50:100,:])
    
    #Fits the explainer
    explainer = shap.Explainer(best_model.predict, X_test.iloc[50:150,:])
    #Calculates the SHAP values
    shap_values = explainer(X_test.iloc[50:100,:])

    #plotting values
    shap.plots.bar(shap_values)
    #Violin plot
    shap.summary_plot(shap_values)
    
    '''Scenario 2: Dropping arrival information and retraining for 24 hour scenario'''
    #Dropping arrival variables
    to_drop = ['ArrDelayMinutes', 'ArrivalDelayGroups', 'ArrTime']
    X_train2 = X_train.drop(columns = to_drop)
    X_test2 = X_test.drop(columns = to_drop)
        
    #Creating subset for hyperparameter tuning
    x_train2_small = X_train2.iloc[:100000, :]
    x_test2_small = X_test2.iloc[:100000, :]
    
    #Re-Fitting Regressor and tree models without arrival information
    regressor_model_24h, regressor_accuracy_24h = fit_regressor(X_train2, y_train, X_test2, y_test)
    tree_model_24h, tree_accuracy_24h = fit_tree(X_train2, y_train, X_test2, y_test)
    
    #Graphing feature importance for random forest and logistic regression
    sklearn_feature_importance(tree_model_24h, regressor_model_24h)
    
    #Training neural network
    models_long = [create_baseline_small(x_train2_small), create_baseline_medium(x_train2_small), create_baseline_large(x_train2_small)]
    names = ['Small', 'Medium', 'Large']
    accuracy_long = []
    histories_long = {}

    #Training models on a subset of data
    for i in range(len(models_long)):
        history_long = models_long[i].fit(x_train2_small, y_train_small, epochs = 25, verbose = 1,
                    validation_data = (x_test2_small, y_test_small))
        accuracy_long.append(models_long[i].evaluate(x_test2_small, y_test_small))
    
        histories_long[i] = {'Results' :pd.DataFrame(history_long.history),
                    'Model' : history_long}
        
    #Visualizing neural network training
    visualize_f1(histories_long)
    
    #Returning best model trained on all data
    best_model_long_index = select_best_model_f1(histories_long)
    best_model_long = models_long[best_model_long_index]
    best_model_long.fit(X_train2, y_train, epochs = 25, verbose = 1,
                            validation_data = (X_test2, y_test))
        
    #Making predictions with all models and putting into dictionary
    reg_predictions_24h = np.round(regressor_model_24h.predict(X_test2),0)
    tree_predictions_24h = np.round(tree_model_24h.predict(X_test2),0)
    nn_predictions_24h = np.round(best_model_long.predict(X_test2),0).astype(int)
    cf_matrix = {'Logistic Classifier': confusion_matrix(y_test, reg_predictions_24h, normalize = 'all') ,
                "RandomForest Clasifier": confusion_matrix(y_test, tree_predictions_24h ,normalize = 'all'),
                'Neural Network':  confusion_matrix(y_test, nn_predictions_24h, normalize = 'all')}
    
    #Generating confusion matrice's for all models predictions
    confusion_matrices(cf_matrix)
    
    #Visualizing feature importance of logistic regression and random forest models, 24h scenario
    sklearn_feature_importance(tree_model, regressor_model)
    neural_network_feature_importance(nn_model, X_test)
        
    # Calculalting F1 Scores 24 hours in advance
    reg_24_hour_f1 = f1_score(y_test, reg_prediction_24h, zero_division=1)
    tree_24_hour_f1 = f1_score(y_test_small, tree_prediction_24h, zero_division=1)
    print("Logistic Regressor F1: ", reg_24_hour_f1)
    print("RF classifier F1: ", tree_24_hour_f1)
    
    #Visualing feature importance of neural network
    #Fits the explainer
    explainer = shap.Explainer(best_model_long.predict, X_test2.iloc[50:150,:])
    #Calculates the SHAP values
    shap_values = explainer(X_test2.iloc[50:150,:])

    #plotting values
    shap.plots.bar(shap_values)
    #Violin plot
    shap.summary_plot(shap_values)
    
    
    
    
    
    
