# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.astype({
    'RAD':float,
    'TAX':float})
    mean_value1 = df['CRIM'].mean()
    mean_value2 = df['ZN'].mean()
    mean_value3 = df['INDUS'].mean()
    mean_value4 = df['CHAS'].mean()
    mean_value5 = df['AGE'].mean()
    mean_value6 = df['LSTAT'].mean()
    df['CRIM'].fillna(value=mean_value1, inplace=True)
    df['ZN'].fillna(value=mean_value2, inplace=True)
    df['INDUS'].fillna(value=mean_value3, inplace=True)
    df['CHAS'].fillna(value=mean_value4, inplace=True)
    df['AGE'].fillna(value=mean_value5, inplace=True)
    df['LSTAT'].fillna(value=mean_value6, inplace=True)
    
    
    X = df.drop('MEDV', axis=1).values 
    y = df['MEDV'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
