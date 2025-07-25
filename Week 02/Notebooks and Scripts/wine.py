import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



def load_data_and_Feature_engineer():
    df=pd.read_csv('D:/Downloads/Week/Week 2/WineQT.csv')
    df=df.drop('Id',axis=1)
    df.drop_duplicates(inplace=True)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df['quality_class'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    
    df=df.drop(['quality'],axis=1)
    return df

def removing_outliers(df):
     columns= df.drop(['quality_class'],axis=1)
     for col in columns:
       Q1 = df[col].quantile(0.25)
       Q3 = df[col].quantile(0.75)

       IQR = Q3 - Q1
       lower_bound = Q1 - 1 * IQR
       upper_bound = Q3 + 1* IQR
       df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
     return df
def encoding(df):

      le = LabelEncoder()
      df['quality_class'] = le.fit_transform(df['quality_class'])
      return df

def split_data(df):
    X=df[['volatile_acidity', 'citric_acid','chlorides', 'density', 'sulphates','alcohol','total_sulfur_dioxide']]
    y=df['quality_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train,X_test,y_train,y_test
def create_pipelines(X_train,X_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    pipeline={
    'KNN':Pipeline([
        ('scalar',StandardScaler()),
        ('classifier',KNeighborsClassifier())
    ]),
    'Decsion tree':Pipeline([
        ('scalar',StandardScaler()),
        ('classifier',DecisionTreeClassifier())
    
    ]),
    'Naive Byes':Pipeline([
        ('scalar',StandardScaler()),
        ('classifier',GaussianNB())
    ]),
    'Random Forest':Pipeline([
        ('scalar',StandardScaler()),
        ('classifier',RandomForestClassifier())
    ]),
    'Lasso (LogReg)': Pipeline([
    ('scalar', StandardScaler()),
    ('classifier', LogisticRegression())
        ]),
}

    return  pipeline
def get_param_grids():
    return {
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance']
        },
        'Decsion tree': {
            'classifier__max_depth': [3, 5, 10],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__splitter': ['best', 'random']
        },
        'Naive Byes': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        'Lasso (LogReg)': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'classifier__max_depth': [3, 5, 7, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }
    }
from sklearn.model_selection import GridSearchCV

def fit_and_evaluate(pipelines, X_train, X_test, y_train, y_test):
    param_grids = get_param_grids()
    best_accuracy = 0
    best_model = None
    best_model_name = None

    for name in pipelines:
        print(f"\nTraining and tuning: {name}")
        model = pipelines[name]
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        Train_pred = best_estimator.predict(X_train)
        Test_pred = best_estimator.predict(X_test)

        Train_accuracy = accuracy_score(y_train, Train_pred)
        Test_accuracy = accuracy_score(y_test, Test_pred)

        print('Best Params:', grid.best_params_)
        print('Training Accuracy:', Train_accuracy)
        print('Testing Accuracy:', Test_accuracy)

        if Test_accuracy > best_accuracy:
            best_accuracy = Test_accuracy
            best_model = best_estimator
            best_model_name = name

    print("\nBest Model:")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    joblib.dump(best_model, "wine_model.joblib")
    with open("model_name.txt", "w") as f:
        f.write(best_model_name)
def main():
    df=load_data_and_Feature_engineer()
    df=removing_outliers(df)
    df=encoding(df)
    X_train,X_test,y_train,y_test=split_data(df)
    pipeline=create_pipelines(X_train,X_test,y_train,y_test)
    fit_and_evaluate(pipeline, X_train,X_test,y_train,y_test)
   
  
main()

