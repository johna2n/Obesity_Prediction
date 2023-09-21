from bed_reader import open_bed, sample_file

from bed_reader import to_bed, tmp_path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


"""## Read genotype-phenotype data after QC"""

X_train_init = pd.read_csv('X_train.csv').set_index('sample')
y_train = pd.read_csv('y_train.csv').replace([1,2], [0, 1])['Phenotype']
X_test_init = pd.read_csv('X_test.csv').set_index('sample')
y_test = pd.read_csv('y_test.csv').replace([1,2], [0, 1])['Phenotype']

"""## Stratification"""

def choose_K(df):
    scaler = StandardScaler()
    df_scl = scaler.fit_transform(df)
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=20, max_iter=300,
            tol=1e-04, random_state=422
        )
        km.fit(df_scl)
        distortions.append(km.inertia_)

    # plot
    font = {'weight' : 'bold','size': 10}

    plt.rc('font', **font)
    k_c = plt.figure(figsize=(10, 6))
    plt.plot([str(i+1) for i in range(10)], distortions, marker='o')
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('Distortion', fontsize=12)
    plt.show()
    k_c.savefig('k_means_clustering.png')

choose_K(X_train_init.iloc[:, 0:-1])

# PCA reduced
def get_pca (X_train, X_test, k):
  pca = PCA(n_components=k, random_state = 0)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)

  var_exp = pca.explained_variance_ratio_.cumsum()
  var_exp = var_exp*100
  plt.bar(range(k), var_exp);
  return (X_train_pca, X_test_pca)

k=5

train_pca, test_pca = get_pca (X_train_init.iloc[:, 0:-1], X_test_init.iloc[:, 0:-1], k)

train_pca = pd.DataFrame(train_pca, columns = ["PC" + str(i) for i in range(k)])
test_pca = pd.DataFrame(test_pca, columns = ["PC" + str(i) for i in range(k)])

X_train = pd.concat([X_train_init.reset_index(drop=True), train_pca.reset_index(drop=True)], axis = 1)
X_test = pd.concat([X_test_init.reset_index(drop=True), test_pca.reset_index(drop=True)], axis = 1)

X_train_after = X_train
X_test_after = X_test

feature_names = list(X_train_after.columns)
len_features = len(feature_names)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert all to numpy
#X_train = X_train.to_numpy()
#X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Training model
def lg_train_data (X_train, X_test, y_train, y_test):

  # Create logistic regression cross validation

  grid = {
        'C': np.power(10.0, np.arange(-5, 5)),
         'solver': ['newton-cg'],
          'random_state': [0]
        }
  
  clf = LogisticRegression(penalty='l2', random_state=0, max_iter=10000, n_jobs = 10)
  lg_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5)
  
  # Train the regressor
  lg_grid.fit(X_train, y_train)
  # Make predictions using the optimised parameters
  lg_pred = lg_grid.predict(X_test)

  
  # Find scores 
  acr = accuracy_score(y_test, lg_pred)
  f1 = f1_score (y_test, lg_pred)
  pre_score = precision_score(y_test, lg_pred)
  rc_score = recall_score (y_test, lg_pred)

  roc_auc = round(roc_auc_score (y_test, lg_pred), 3)

  print("accuracy:", acr,  "precision_score:", 
        pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)

  best_prs = lg_grid.best_estimator_

  print("Best Parameters:\n", best_prs)
  print("Best Score:\n", 'roc_auc:', roc_auc)
  # Get coefficients of the model 
  coef = pd.DataFrame(lg_grid.coef_.T, index = feature_names)
  var = coef[coef[0] != 0]
  print("Head of coefficients", coef.sort_values(by = 0, ascending=False).head())
  print("ElasticNet picked " + str(sum(coef[0] != 0)) + " variables and eliminated the other " +  str(sum(coef[0] == 0)) + " variables")
  pd.DataFrame({'variable':var}).to_csv('log_features.csv')
  return (acr, pre_score, rc_score, f1, roc_auc)
  
acr, pre_score, rc_score, f1, roc_auc = lg_train_data (X_train, X_test, y_train, y_test)
