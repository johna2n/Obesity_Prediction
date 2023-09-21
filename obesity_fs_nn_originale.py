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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x,y):
        super().__init__()
        self.x = x
        self.y = y 
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

# define the train and test dataloader
train_loader = torch.utils.data.DataLoader(Dataset(X_train, y_train))
test_loader = torch.utils.data.DataLoader(Dataset(X_test, y_test))

"""## Build a Baseline Model"""

torch.manual_seed(1)

# code a neural network with the nn module imported into the class
class Obesity_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(len_features, 280)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(280,560)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(560,2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        lin1_out = self.linear1(x)
        sigmoid1_out = self.sigmoid1(lin1_out)
        lin2_out = self.linear2(sigmoid1_out)
        sigmoid2_out = self.sigmoid2(lin2_out)
        lin3_out = self.linear3(sigmoid2_out)
        softmax_out = self.softmax(lin3_out)
        return softmax_out

model = Obesity_Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
total_loss, total_acc = list(),list()
feat_imp = np.zeros(X_train.shape[1])
num_epochs = 500
for epoch in range(num_epochs):
    losses = 0 
    for idx, (x,y) in enumerate(train_loader):
        x,y = x.float(), y.type(torch.LongTensor)
        x.requires_grad=True
        optimizer.zero_grad()
        # Check if the programa can be run with model(x) and model.forward()
        preds=model.forward(x)
        loss=criterion(preds,y)
        x.requires_grad = False
        loss.backward()
        optimizer.step()
        losses+=loss.item()
    total_loss.append(losses/len(train_loader))
    if epoch%20==0:
        print("Epoch:", str(epoch+1), "\tLoss:", total_loss[-1])

# Save the model
torch.save(model.state_dict(), 'obesity_model.pt')

model.eval()
correct=0

with torch.no_grad():
    y_pred = []
    y_obs = []

    for idx, (x,y) in enumerate(test_loader):
    
        x,y = x.float(), y.type(torch.LongTensor)
        pred = model(x)
        preds_class = torch.argmax(pred)
        y_pred.append(preds_class.numpy())
        y_obs.append(y.numpy()[0])
    
    # Find scores 
    acr = accuracy_score(y_obs, y_pred)
    f1 = f1_score (y_obs, y_pred)
    pre_score = precision_score(y_obs, y_pred)
    rc_score = recall_score (y_obs, y_pred)
    pre, recall, _ = precision_recall_curve(y_obs, y_pred)
    roc_auc = roc_auc_score (y_obs, y_pred)

print("accuracy:", acr,  "precision_score:", 
    pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)

test_input_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)

"""## Calculate the Integrated Gradients"""

ig = IntegratedGradients(model)

test_input_tensor.requires_grad_()
attr, delta = ig.attribute(test_input_tensor, target = 1, return_convergence_delta  = True)
attr = attr.detach().numpy()

# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        importance = plt.figure(figsize=(12,6), dpi=150)
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True, rotation=90, fontsize=4)
        plt.tick_params(axis='x', pad=15)
        plt.xlabel(axis_title)
        plt.title(title)
        importance.savefig('feature_importance.png')
visualize_importances(feature_names, np.mean(np.abs(attr), axis=0))

"""## Get Top K Least Important Features and Retrain"""

def make_new_dataset(features_to_be_dropped):
    
    X_train_drop = X_train_after.drop(features_to_be_dropped, axis=1)
    X_test_drop = X_test_after.drop(features_to_be_dropped, axis=1)

    scaler = StandardScaler()
    X_train_drop = scaler.fit_transform(X_train_drop)
    X_test_drop = scaler.transform(X_test_drop)

    # define the train and test dataloader
    train_loader = torch.utils.data.DataLoader(Dataset(X_train_drop, y_train))
    test_loader = torch.utils.data.DataLoader(Dataset(X_test_drop, y_test))
    
    return train_loader,test_loader

def define_model(k_features):
    torch.manual_seed(1)
    # code a neural network with the nn module imported into the class
    class Obesity_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(len_features-k_features, 280) # since features have been dropped chaneg input layer
            self.sigmoid1 = nn.Sigmoid()
            self.linear2 = nn.Linear(280, 560)
            self.sigmoid2 = nn.Sigmoid()
            self.linear3 = nn.Linear(560, 2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self,x):
            lin1_out = self.linear1(x)
            sigmoid1_out = self.sigmoid1(lin1_out)
            lin2_out = self.linear2(sigmoid1_out)
            sigmoid2_out = self.sigmoid2(lin2_out)
            lin3_out = self.linear3(sigmoid2_out)
            softmax_out = self.softmax(lin3_out)
            return softmax_out
    return Obesity_Model

def train_model(Obesity_Model):
    model = Obesity_Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss, total_acc = list(),list()

    num_epochs = 500

    for epoch in range(num_epochs):
        losses = 0 
        for idx, (x,y) in enumerate(train_loader):
            x,y = x.float(), y.type(torch.LongTensor)
            x.requires_grad=True
            optimizer.zero_grad()
            # check if the progrma can be run with model(x) and model.forward()
            preds=model.forward(x)
            loss=criterion(preds,y)
            x.requires_grad = False
            loss.backward()
            optimizer.step()
            losses+=loss.item()
        total_loss.append(losses/len(train_loader))
        if epoch%50==0:
            print("Epoch:", str(epoch+1), "\tLoss:", total_loss[-1])
    return model

def test_results(model, test_loader):
    model.eval()
    y_pred = []
    y_obs = []

    for idx, (x,y) in enumerate(test_loader):
        with torch.no_grad():
            x,y = x.float(), y.type(torch.LongTensor)
            pred = model(x)
            preds_class = torch.argmax(pred)
            y_pred.append(preds_class.numpy())
            y_obs.append(y.numpy()[0])
    # Find scores 
    acr = accuracy_score(y_obs, y_pred)
    f1 = f1_score (y_obs, y_pred)
    pre_score = precision_score(y_obs, y_pred)
    rc_score = recall_score (y_obs, y_pred)
    pre, recall, _ = precision_recall_curve(y_obs, y_pred)
    roc_auc = round(roc_auc_score (y_obs, y_pred), 3)
        
    print("accuracy:", acr,  "precision_score:", 
        pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)
    
    return (acr, pre_score, rc_score, f1, roc_auc)

feat_imp=  np.mean(np.abs(attr), axis=0)
[(a,b) for (a,b) in sorted(zip(feat_imp,feature_names))]

best_accuracy = list()
all_scores = []
for i in range(0, 141):
    k_features=i
    features_to_be_dropped = [b for (a,b) in sorted(zip(feat_imp,feature_names))][0:k_features]
    #print(features_to_be_dropped)
    train_loader, test_loader = make_new_dataset(features_to_be_dropped)
    Obesity_Model = define_model(k_features)
    trained_model = train_model(Obesity_Model)
    acr, pre_score, rc_score, f1, roc_auc = test_results(trained_model,test_loader)
    best_accuracy.append(acr)
    all_scores.append([acr, pre_score, rc_score, f1, roc_auc])

print(max(best_accuracy))
id = np.argmax(best_accuracy)
results = all_scores[id]
print(results)

print (id)

print('Significant features')
ft = [(a,b) for (a,b) in sorted(zip(feat_imp,feature_names))]
#print (ft)
pd.DataFrame({'IFeatures':ft}).to_csv('IFeat.csv')
pd.DataFrame({'eval':best_accuracy}).to_csv('eval.csv')
pd.DataFrame({'all_score':all_scores}).to_csv('all_scores.csv')