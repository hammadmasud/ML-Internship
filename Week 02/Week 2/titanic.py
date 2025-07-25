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

def load_data_and_Feature_engineer():
    df=pd.read_csv('D:/Downloads/Week/Week 2/Titanic-Dataset.csv')
    df['family_size']=df['Parch']+df['SibSp']+1
    df=df.drop('Parch',axis=1)
    df=df.drop('SibSp',axis=1)
    df['Age']=pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Aged'])
    df['Cabin1'] = df['Cabin'].str[0] 
    df['HasCabin'] = (df['Cabin1'] != 'U').astype(int)
    return df
def missing_values(df):
     df['Cabin']=df['Cabin'].fillna('U')
     Embarked_mode=df['Embarked'].mode()[0]
     df['Embarked']=df['Embarked'].fillna(Embarked_mode)
     age_mode=df['Age'].mode()[0]
     df['Age']=df['Age'].fillna(age_mode)

     return df
def encoding(df):
     df = pd.get_dummies(df, columns=['Age'], prefix='Age', drop_first=False)
     df = pd.get_dummies(df, columns=['Sex'], prefix='Sex', drop_first=False)
     df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=False)
     return df
def removing_outliers(df):
     column1=['Fare','family_size','Fare']
     for col in column1:
       Q1 = df[col].quantile(0.25)
       Q3 = df[col].quantile(0.75)
       IQR = Q3 - Q1
       lower_bound = Q1 - 1.5 * IQR
       upper_bound = Q3 + 1.5 * IQR
       df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
     return df
def split_data(df):
    y=df['Survived']
    X=df[['Pclass','Fare','family_size','Age_Child','Age_Teen','Age_Adult','Age_Aged','Sex_female','Sex_male'	,'Embarked_C','Embarked_Q','Embarked_S','HasCabin']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test
def create_pipelines(X_train,X_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    pipeline={
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
    return  pipeline
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
 joblib.dump(best_model, "titanic_model.joblib")
 with open("model_name.txt", "w") as f:
    f.write(best_model_name)

def main():
    df=load_data_and_Feature_engineer()
    df=missing_values(df)
    df=encoding(df)
    df=removing_outliers(df)
    X_train,X_test,y_train,y_test=split_data(df)
    pipeline=create_pipelines(X_train,X_test,y_train,y_test)
    fit_and_evaluate(pipeline, X_train,X_test,y_train,y_test)
  
main()
