import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
def load_and_prepredata():
    iris=load_iris()
    iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
    iris_data['target']=iris.target

    iris_data.drop_duplicates(inplace=True)
    X=iris_data[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
    y=iris_data['target']
    return X,y
def scale_features(X):
    scalar=StandardScaler()
    return scalar.fit_transform(X),scalar
def encode_features(y):
    encoder=LabelEncoder()
    return encoder.fit_transform(y)
def train_model(X_train,y_train):
    Model=LogisticRegression()
    Model.fit(X_train,y_train)
    return Model
def evaluate_the_model(model,X_test,y_test):
    y_pred=model.predict(X_test)
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4) )
    print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, average='weighted'), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred, average='weighted'), 4))
       

def main():
    X, y = load_and_prepredata()
    X_scaled ,scaler= scale_features(X)
    y_encoded =encode_features(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_the_model(model, X_test, y_test)
    joblib.dump(model, 'iris_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

if __name__ == "__main__":
    main()