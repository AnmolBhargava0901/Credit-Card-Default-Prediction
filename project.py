import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC #support vector machine classifier
from sklearn.neural_network import MLPClassifier #multi layer perceptron classifier or neural network

data = pd.read_csv("C:\\Users\\anmol\\Downloads\\archive\\UCI_Credit_Card.csv")
print(data)
pd.set_option('display.max_columns',None)
print("-----------------DATA-----------------")
print(data)
print("-----------------IFORMATION-----------------")
print(data.info())


def one_hot_encoding(df,column_dict):
    df = df.copy()
    for column,prefix in column_dict.items():  
       dummies = pd.get_dummies(df[column],prefix=prefix)
       df = pd.concat([df,dummies],axis=1)
       df = df.drop(column,axis=1)
    return df

def preprocess_input(df):
    df = df.copy()

    df = df.drop('ID',axis=1)
    df=one_hot_encoding(
    df,{
        'EDUCATION':'EDU',
        'MARRIAGE':'MAR',  
    }
    )

    y = df['default.payment.next.month'].copy()
    X = df.drop('default.payment.next.month',axis=1).copy()

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

    return X,y

X,y = preprocess_input(data)
print("-----------------X-----------------")
print(X)
print("-----------------X.mean()-----------------")
print(X.mean())
print("-----------------y-----------------")
print(y)

corr = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='coolwarm')

plt.title('Correlation heatmap')
plt.show()
unique_values = {column:len(X[column].unique())for column in X.columns}
print("-----------------unique_values-----------------")
print(unique_values)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.7,random_state=42)
models={
    LogisticRegression():"   Logistic Regression",
    SVC():               "Support vector machine",
    MLPClassifier():     "        Neural Network",
}
for model in models.keys(): 
    model.fit(X_train,y_train)
    print(f"{models[model]} trained")
    print("-----------------score-----------------")

    for mode,name in models.items():
        print(name + ":{:.4f}%".format(model.score(X_test, y_test)*100))