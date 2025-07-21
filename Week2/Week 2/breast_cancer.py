import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
def load_data():
    df=pd.read_csv('D:/Downloads/Week/Week 2/breast-cancer.csv')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df 
   
   
def remove_outliers(df):
    
    mask = pd.Series(True, index=df.index)
    columns=['concave points_worst','perimeter_worst','concave points_mean','radius_worst','perimeter_mean','area_worst','radius_mean','area_mean','concavity_mean','concavity_worst','compactness_mean','compactness_worst','radius_se','perimeter_se','area_se','texture_worst','smoothness_worst','symmetry_worst','texture_mean','concave points_se','smoothness_mean','symmetry_mean']
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
    return columns
def prepare_data(df,columns):
    
     X=df[columns]
     y=df['diagnosis']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
     return X_train,X_test,y_train,y_test
def create_pipelines(X_train,X_test,y_train,y_test):
     pipelines={
    'KNN':Pipeline([
        ('scalar',StandardScaler()),
        ('classifier',KNeighborsClassifier(n_neighbors=3))
    ]),
    'Decsion tree':Pipeline([
        ('scalar',StandardScaler()),
        ('classifier',DecisionTreeClassifier(max_depth=5,random_state=42, splitter='random',criterion='entropy'))
    
    ]),
    'Naive Byes':Pipeline([
        ('scalar',StandardScaler()),
        ('classifier',GaussianNB(var_smoothing=1e-8))
    ])
    }
     return pipelines
def fit_and_evaluate(pipelines,X_train,X_test,y_train,y_test):
 best_accuracy=0
 best_model=None
 best_model_name=None
 for name,model in pipelines.items():
    print(name)
    model.fit(X_train,y_train)
    Train_pred=model.predict(X_train)
    Test_pred=model.predict(X_test)
    Train_accuracy=accuracy_score(Train_pred,y_train)
    Test_accuracy=accuracy_score(Test_pred,y_test)
    print('Training Accuracy:', Train_accuracy)
    print('Testing Accuracy:', Test_accuracy)
   
    if Test_accuracy > best_accuracy:
            best_accuracy = Test_accuracy
            best_model = model
            best_model_name = name

 print("\nBest Model:")
 print(f"Model: {best_model_name}")
 print(f"Accuracy: {best_accuracy:.4f}")      
 joblib.dump(best_model, "breast_cancer.joblib")
 with open("cancer_model_name.txt", "w") as f:
    f.write(best_model_name)
    
 

def main():
    df=load_data()
    columns=remove_outliers(df)
    X_train,X_test,y_train,y_test=prepare_data(df,columns)
    pipelines=create_pipelines(X_train,X_test,y_train,y_test)

    fit_and_evaluate(pipelines,X_train,X_test,y_train,y_test)
main()