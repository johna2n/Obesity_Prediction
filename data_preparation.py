
import pandas as pd
import numpy as np
import stratification.pca as pca

if __name__ == '__main__':

    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----                DATA PREPARATION              -----           |")
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
    X_test_init = pd.read_csv('data/X_test.csv').set_index('sample')
    
    print("")
    
    # Choose 5 principle components
    k=5

    train_pca, test_pca = pca.get_pca (X_train_init.iloc[:, 0:-1], X_test_init.iloc[:, 0:-1], k)

    train_pca = pd.DataFrame(train_pca, columns = ["PC" + str(i) for i in range(k)])
    test_pca = pd.DataFrame(test_pca, columns = ["PC" + str(i) for i in range(k)])

    X_train = pd.concat([X_train_init.reset_index(drop=True), train_pca.reset_index(drop=True)], axis = 1)
    X_test = pd.concat([X_test_init.reset_index(drop=True), test_pca.reset_index(drop=True)], axis = 1)
    
    X_train.to_csv('X_train_stf.csv', index = None)
    X_test.to_csv('X_test_stf.csv', index= None)
    
    print("")
    print("********************************* FINISHED *********************************")
    print("")

