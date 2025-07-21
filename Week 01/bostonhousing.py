import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    df=pd.read_csv('D:\Downloads\Week\HousingData.csv')
    return df

def remove_missing_values(df):
    lstat_median=df['LSTAT'].median()
    df['LSTAT']=df['LSTAT'].fillna(lstat_median)
    CRIM_median=df['CRIM'].median()
    df['CRIM']=df['CRIM'].fillna(CRIM_median)
    ZN_median=df['ZN'].median()
    df['ZN']=df['ZN'].fillna(ZN_median)
    INDUS_mean=df['INDUS'].mean()
    df['INDUS']=df['INDUS'].fillna(INDUS_mean)
    CHAS_mode=df['CHAS'].mode()[0]
    df['CHAS']=df['CHAS'].fillna(CHAS_mode)
    AGE_median=df['AGE'].median()
    df['AGE']=df['AGE'].fillna(AGE_median)

    return df
def remove_outliers(df):
    columns = [ 'LSTAT', 'RM','PTRATIO','CRIM','INDUS','AGE','TAX']
    for col in columns:
       Q1 = df[col].quantile(0.25)
       Q3 = df[col].quantile(0.75)
       IQR = Q3 - Q1
       lower_bound = Q1 - 1.5 * IQR
       upper_bound = Q3 + 1.5 * IQR
       df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
def split_data(df):
    X=df[['RM','LSTAT','PTRATIO','CRIM','AGE','TAX','INDUS']]
    y=df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test
def train_model(X_train,y_train ):
     pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('regressor', LinearRegression())
      ])
     pipeline.fit(X_train, y_train)
     return pipeline

       
    

def main():
    feature_columns = [['RM','LSTAT','PTRATIO','CRIM','AGE','TAX','INDUS']]
    df=load_data()
    df = remove_missing_values(df)
    df = remove_outliers(df)
    X_train,X_test,y_train ,y_test = split_data(df)
    pipeline  = train_model( X_train,y_train)
 


    joblib.dump(pipeline, 'boston_pipeline.joblib')
    joblib.dump(feature_columns, 'feature_names.joblib')


    print("Model Saved")

main()
