# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer
from scipy import stats
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import stratification.pca as pca
import models.nn as nn

if __name__ == '__main__':
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----        NEURAL NETWORK FEATURE SELECTION       -----          |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")
    
    # Read genotype-phenotype data after subsequent data preprocessing
    X_train_init = pd.read_csv('data/X_train.csv').set_index('sample')
    y_train = pd.read_csv('data/y_train.csv').replace([1,2], [0, 1])['Phenotype']
    X_test_init = pd.read_csv('data/X_test.csv').set_index('sample')
    y_test = pd.read_csv('data/y_test.csv').replace([1,2], [0, 1])['Phenotype']
    print("")
    
    # Choose 5 principle components
    k=5

    train_pca, test_pca = pca.get_pca (X_train_init.iloc[:, 0:-1], X_test_init.iloc[:, 0:-1], k)

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
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    
    # Define the train and test dataloader
    train_loader = torch.utils.data.DataLoader(nn.Dataset(X_train, y_train))
    test_loader = torch.utils.data.DataLoader(nn.Dataset(X_test, y_test))

    print ("Start to train models")
    # Build a Baseline Model

    torch.manual_seed(1)

    model = nn.Obesity_Model()
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
    print("Save the model")

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
    
    print("Calculate the Integrated Gradients")
    # Calculate the Integrated Gradients
    ig = IntegratedGradients(model)

    test_input_tensor.requires_grad_()
    attr, delta = ig.attribute(test_input_tensor, target = 1, return_convergence_delta  = True)
    attr = attr.detach().numpy()


    nn.visualize_importances(feature_names, np.mean(np.abs(attr), axis=0))

    feat_imp =  np.mean(np.abs(attr), axis=0)
    [(a,b) for (a,b) in sorted(zip(feat_imp,feature_names))]

    best_accuracy = list()
    all_scores = []

    print("Find the best accuracy")
    for i in range(0, 141):
        k_features=i
        print("For ",k_features, " features" )
        features_to_be_dropped = [b for (a,b) in sorted(zip(feat_imp,feature_names))][0:k_features]
        train_loader, test_loader = nn.make_new_dataset(features_to_be_dropped, X_train_after, X_test_after, y_train, y_test)
        Obesity_Model = nn.define_model(k_features, len_features)
        trained_model = nn.train_model(Obesity_Model, train_loader)
        acr, pre_score, rc_score, f1, roc_auc = nn.test_results(trained_model,test_loader)
        best_accuracy.append(acr)
        all_scores.append([acr, pre_score, rc_score, f1, roc_auc])

    print("The best accuracy:", max(best_accuracy))
    id = np.argmax(best_accuracy)
    results = all_scores[id]
    print(results)

    print (id)

    print('Significant features')
    ft = [(a,b) for (a,b) in sorted(zip(feat_imp,feature_names))]
    print (ft)

    print("********************************** SAVING **********************************")
    pd.DataFrame({'IFeatures':ft}).to_csv('IFeat.csv')
    pd.DataFrame({'eval':best_accuracy}).to_csv('eval.csv')
    pd.DataFrame({'all_score':all_scores}).to_csv('all_scores.csv')

    print("")
    print("********************************* FINISHED *********************************")
    print("")
    