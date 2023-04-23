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
        
    #One Hot Encoding State name
    state_encode = pd.get_dummies(df_sample.OriginStateName, prefix='State')
        
    #Adding encoded variables to sample df
    df_sample = pd.concat([df_sample, state_encode], axis =1)
    y = df_sample['Delay']
    x = df_sample.drop(columns = ['Delay'])
    
    return x, y
    
def normalize_df( numeric_df):
        
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


def train_neural_network(input_df, hidden_layers = 2):

    input_features = input_df.shape[1]
    model = Sequential()
    model.add(Dense(input_features, input_shape=(input_features,), activation='relu'))
    for layer in len(range(hidden_layers)):
        
        model.add(Dropout(.2, input_shape=(input_features,))) #Dropout layer to avoid overfitting
        model.add(Dense(input_features, input_shape=(input_features*layer,), activation='relu')) #Triangular structure
        model.add(Dropout(.2, input_shape=(input_features,))) #Dropout layer to avoid overfitting
    #Adding Output Layer
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
    
if __name__ == 'main':
    
    #Load dataframe, sort by time, remove columns
    df = pd.read_csv('C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Masters Acquisition\\Capstone\\Data for Modeling\\Medium Sample, df 4M.csv')
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
    
    #Training neural network
    nn_model = train_neural_network(X_train, hidden_layers = 2)
    
    #Making predictions with all models and putting into dictionary
    reg_predictions = np.round(regressor_model.predict(X_test),0)
    tree_predictions = np.round(tree_model.predict(X_test),0)
    nn_predictions = np.round(nn_model.predict(X_test),0).astype(int)
    cf_matrix = {'Logistic Classifier': confusion_matrix(y_test, reg_predictions, normalize = 'all') ,
                "RandomForest Clasifier": confusion_matrix(y_test, tree_predictions,normalize = 'all'),
                'Neural Network': confusion_matrix(y_test, nn_predictions, normalize = 'all')}
    
    #Generating confusion matrice's for all models predictions
    confusion_matrices(cf_matrix)
    
    #Visualizing feature importance
    sklearn_feature_importance(tree_model, regressor_model)
    neural_network_feature_importance(nn_model, X_test)
    
    
    '''Scenario 2: Dropping arrival information and retraining for 24 hour scenario'''
    #Dropping arrival variables
    to_drop = ['ArrDelayMinutes', 'ArrivalDelayGroups', 'ArrTime']
    X_train2 = X_train.drop(columns = to_drop)
    X_test2 = X_test.drop(columns = to_drop)
    
    #Re-Fitting Regressor and tree models without arrival information
    regressor_model_24h, regressor_accuracy_24h = fit_regressor(X_train2, y_train, X_test2, y_test)
    tree_model_24h, tree_accuracy_24h = fit_tree(X_train2, y_train, X_test2, y_test)
    
    #Graphing feature importance for random forest and logistic regression
    sklearn_feature_importance(tree_model_24h, regressor_model_24h)
    
    #Training neural network
    nn_model_24h = train_neural_network(X_train2, hidden_layers = 2)
    
    #Making predictions with all models and putting into dictionary
    reg_predictions_24h = np.round(regressor_model_24h.predict(X_test2),0)
    tree_predictions_24h = np.round(tree_model_24h.predict(X_test2),0)
    nn_predictions_24h = np.round(nn_model_24h.predict(X_test2),0).astype(int)
    cf_matrix = {'Logistic Classifier': confusion_matrix(y_test, reg_predictions_24h, normalize = 'all') ,
                "RandomForest Clasifier": confusion_matrix(y_test, tree_predictions_24h ,normalize = 'all'),
                'Neural Network':  confusion_matrix(y_test, nn_predictions_24h, normalize = 'all')}
    
    #Generating confusion matrice's for all models predictions
    confusion_matrices(cf_matrix)
    
    #Visualizing feature importance
    sklearn_feature_importance(tree_model, regressor_model)
    neural_network_feature_importance(nn_model, X_test)
    
    
    
    
    
    
    