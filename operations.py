import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def append_DF(C1, C2):
    c1_c2 = C1.append(C2, ignore_index=True, verify_integrity=True, sort=True)
    return c1_c2


def split_features(DF,X1, X2, Y):
    X1 = DF.iloc[:,X1]; X1 = X1.to_numpy()
    X2 = DF.iloc[:,X2]; X2 = X2.to_numpy()
    Y = DF.iloc[:,Y]; Y = Y.to_numpy()
    X1_train = X1[:30]
    X1_test = X1[30:50]
    X2_train = X2[:30]
    X2_test = X2[30:50]
    Y_train = Y[:30]
    Y_test = Y[30:50]
    train_data = pd.DataFrame(columns=[X1_train, X2_train,Y_train])
    test_data = pd.DataFrame(columns=[X1_test, X2_test,Y_test])
    return train_data,test_data


def visualization(X1, X2, DF):
    sns.scatterplot(x=X1, y=X2, data=DF, hue='species'
                    , palette={0: 'red', 1: 'green', 2: 'blue'})
    plt.show()