"""
Diplomski, data preprocessing
@author: EmaPajic
"""

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import OneHotEncoder

def preprocess(check_rf_val = False, save = True):
    df = pd.read_csv(r'data\data_spam.csv')
    
    print(df.shape)
    Y = np.array(df['Column58'])

    df.drop('Column58', axis=1, inplace = True)
    X = np.array(df)
    
    print(X.shape, Y.shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    
    if check_rf_val:
        clf = RandomForestClassifier(n_estimators=10, max_depth=10)
        clf.fit(X_train, Y_train)
        Y_test_pred = clf.predict_proba(X_test)
        #print(Y_test_pred[1])
        #print(log_loss(Y_test, Y_test_pred))
        #print(np.argmax(Y_test_pred, axis = 1))
        print(accuracy_score(Y_test, np.argmax(Y_test_pred, axis = 1)))
    
    if save:
        df_train_x = pd.DataFrame(X_train)
        df_train_y = pd.DataFrame(Y_train)
        df_test_x = pd.DataFrame(X_test)
        df_test_y = pd.DataFrame(Y_test)
        
        
        df_train_x.to_csv ('data/train_x_spam.csv', index = False, header = False)
        df_train_y.to_csv ('data/train_y_spam.csv', index = False, header = False)
        df_test_x.to_csv ('data/test_x_spam.csv', index = False, header = False)
        df_test_y.to_csv ('data/test_y_spam.csv', index = False, header = False)
    
def main():
    preprocess(check_rf_val = True, save = False)
    
if __name__ == '__main__':
    main()