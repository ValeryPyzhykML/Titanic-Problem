import numpy as np
import pandas as pd

data = pd.read_csv("D:\ML\Titanic\\train.csv", dtype = {'Cabin': np.str} )
y = data["Survived"]
print(y)


data = data.drop(["Name", "Survived", "PassengerId", "Ticket"], axis=1) # clean unneeded data

data["Sex"] = data["Sex"].map({'male': 0, 'female': 1})

for e in data["Embarked"].dropna().unique():
    data["Embarked-%s"  %  e]  = data["Embarked"].map(lambda x: 1 if x == e else 0)
data = data.drop(["Embarked"], axis=1) # prepared

data["iscabin"] = data["Cabin"].notnull().map(lambda x: 1 if x else 0)
data["isage"] = data["Age"].notnull().map(lambda x: 1 if x else 0)

data = data.drop(["Cabin"], axis=1) # TODO investigate

meanAge = data["Age"].mean()
data["Age"] = data["Age"].fillna(meanAge)

print(data["Parch"].unique())
print(data)

w = pd.Series(np.zeros(len(data.columns)+1))  
print(w)



