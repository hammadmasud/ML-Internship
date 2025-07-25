import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
scaler=StandardScaler()
def load_and_prepare_data():
    df = pd.read_csv('D:\Downloads\Week\Mall_Customers.csv')
    df.drop('CustomerID', axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=False)
    return df


def scale_data(df):
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, "scaler_kmeans.pkl") 
    return df_scaled


def apply_pca(df_scaled):

    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(df_scaled)
    joblib.dump(pca, "pca_kmeans.pkl")  
    return df_pca


 
def cluster_and_plot(df_pca, df,n_clusters=4, ):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_pca)
    label_map = {
    0: 'Male- Medium Income -Less Spending',
    1: 'Female - Medium Income -High Spending',
    2: 'Female- Medium Income -Less Spending',
    3: 'Male- Medium Income -High Spending',
    }

    df['cluster_label'] = df['cluster'].map(label_map)
    return df_pca,df

def split_data(df_pca,df):
    X = df_pca  
    y = df['cluster'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
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
   
}

    return  pipeline
def get_param_grids():
    return {
         'KNN': {
        'classifier__n_neighbors': [3, 5, 7, 9,11,13,15],
        'classifier__weights': ['distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']

        },
        'Decsion tree': {
            'classifier__max_depth': [3, 5, 10],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__splitter': ['best', 'random']
        },
        'Naive Byes': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        },
      
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
    joblib.dump(best_model, 'kmeanspsa.joblib')
    

    
   
def main():
    df = load_and_prepare_data()
    df_scaled = scale_data(df)
    df_pca= apply_pca(df_scaled)
    cluster_and_plot( df_pca,df, n_clusters=5)
    X_train,X_test,y_train,y_test=split_data(df_pca,df)
    pipeline=create_pipelines(X_train,X_test,y_train,y_test)
    fit_and_evaluate(pipeline, X_train,X_test,y_train,y_test)

if __name__ == "__main__":
    main()
