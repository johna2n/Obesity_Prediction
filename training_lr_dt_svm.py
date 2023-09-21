# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import stratification.pca as pca
import models.lr as lr
import models.dt as dt
import models.svm as svm
import models.xgboost as xgb

import time

if __name__ == '__main__':

    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----               FEATURE SELECTIONS             -----           |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")

    
    # Read genotype-phenotype data 
    X_train = pd.read_csv('X_train_stf.csv')
    X_test = pd.read_csv('X_test_stf.csv')
    y_train = pd.read_csv('data/y_train.csv').replace([1,2], [0, 1])['Phenotype']
    y_test = pd.read_csv('data/y_test.csv').replace([1,2], [0, 1])['Phenotype']
    feature_names = list(X_train.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert all to numpy
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    '''
    print("Logistic regression")
    start_time = time.time()
    lr.train_lr (X_train, X_test, y_train, y_test, feature_names)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    print("")
    '''
    '''
    print("Decision-Tree RFE")
    # Start timer
    start_time = time.time()
    dt.rfe_dt(X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    '''
 
    print("SVM RFE")
    # Start timer
    start_time = time.time()
    svm.rfe_svc(X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    
    #print("")
    #print("SVM RFE")
    #svm.rfe_svm(X_train, y_train, X_test, y_test)

    print("********************************** SAVING **********************************")
   
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    