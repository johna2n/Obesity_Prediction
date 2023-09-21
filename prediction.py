# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

import stratification.pca as pca
import models.lr as lr
import models.dt as dt
import models.svm as svm
import evaluations.metrics as mt
import evaluations.visualization as vs

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
    
    # Read genotype-phenotype data after subsequent data preprocessing
    X_train = pd.read_csv('X_train_stf.csv')
    X_test = pd.read_csv('X_test_stf.csv')
    y_train = pd.read_csv('data/y_train.csv').replace([1,2], [0, 1])['Phenotype']
    y_test = pd.read_csv('data/y_test.csv').replace([1,2], [0, 1])['Phenotype']
    print("")

    feature_names = list(X_train.columns)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert all to numpy
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    
    print("")
    print("Decision-Tree RFE")
    indice_features = pd.read_csv('dt_features.csv')['features']
    dt_features = [feature_names[i] for i in indice_features]
    pd.DataFrame({'features': dt_features}).to_csv('dt_feature_names.csv')
    # Load and fit model
    dt_model = load("dt.joblib")

    # Get data with the selected features
    X_train_dt = X_train[:, indice_features]
    X_test_dt = X_test[:, indice_features]
    print("Number of Selected Features: ", X_test_dt.shape[1])

    mt.eval(dt_model, X_test_dt, y_test).to_csv('dt_evaluations.csv')
    vs.draw_roc_curve (dt_model, X_test_dt, y_test)
    '''
    print("")
    print("Logistic Regression")
    # Load the Logistic Regression Model
    lg_model = load("lg.joblib")
    mt.eval(lg_model, X_test, y_test).to_csv('lg_evaluations.csv')
    vs.draw_roc_curve (lg_model, X_test, y_test)
    '''

