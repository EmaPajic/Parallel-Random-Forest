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
    df = pd.read_json (r'data\\train.json')
    
    print(df.shape)
    
    
    features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
    
    # count of photos #
    df["num_photos"] = df["photos"].apply(len)
    
    # count of "features" #
    df["num_features"] = df["features"].apply(len)
    
    # count of words present in description column #
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    
    # Let us extract some features like year, month, day, hour from date columns #
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    
    # adding all these new features to use list #
    features_to_use.extend(["num_photos", "num_features",
                        "num_description_words","created_year",
                        "created_month", "created_day", "listing_id",
                        "created_hour"])
    
    categorical = ["display_address", "manager_id", "building_id", "street_address"]
    for f in categorical:
            if df[f].dtype=='object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df[f].values))
                df[f] = lbl.transform(list(df[f].values))
                features_to_use.append(f)
                
    df['features'] = df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    tfidf = CountVectorizer(stop_words='english', max_features = 20)
    tr_sparse = tfidf.fit_transform(df["features"])
    
    X = sparse.hstack([df[features_to_use], tr_sparse]).toarray()
    
    target_num_map = {'high':0, 'medium':1, 'low':2}
    Y = np.array(df['interest_level'].apply(lambda x: target_num_map[x]))
    
    print(X.shape, Y.shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = (5000 / 49352))
    
    if check_rf_val:
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train, Y_train)
        Y_test_pred = clf.predict_proba(X_test)
        print(Y_test_pred[1])
        print(log_loss(Y_test, Y_test_pred))
        print(np.argmax(Y_test_pred, axis = 1))
        print(accuracy_score(Y_test, np.argmax(Y_test_pred, axis = 1)))
    
    if save:
        df_train_x = pd.DataFrame(X_train)
        df_train_y = pd.DataFrame(Y_train)
        df_test_x = pd.DataFrame(X_test)
        df_test_y = pd.DataFrame(Y_test)
        
        
        df_train_x.to_csv ('data/train_x.csv', index = False, header = False)
        df_train_y.to_csv ('data/train_y.csv', index = False, header = False)
        df_test_x.to_csv ('data/test_x.csv', index = False, header = False)
        df_test_y.to_csv ('data/test_y.csv', index = False, header = False)
    
def main():
    preprocess(check_rf_val = False, save = True)
    
if __name__ == '__main__':
    main()