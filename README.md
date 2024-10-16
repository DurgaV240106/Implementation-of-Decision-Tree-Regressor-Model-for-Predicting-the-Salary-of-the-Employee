# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 
2.Upload the dataset and check for any null values using .isnull() function. 
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays. 
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DURGA V
RegisterNumber:  212223230052
*/
```
```

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("/content/Salary (2).csv")
data.head()
```
## Output:
![image](https://github.com/user-attachments/assets/fb54de66-5d6f-4feb-aa71-a06e00ccb87d)

```
data.info()
data.isnull().sum()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/96a6bf7b-29d1-412a-b030-0a7dc2fe95c9)

```

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/8eb06f3b-0034-4079-8d0c-68a965c41261)

```
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/8eb71757-7e9e-48cf-be08-86eb3e07d74b)
```
r2=metrics.r2_score(y_test,y_pred)
r2
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/36652975-933c-40f1-9667-f333f15efcc1)
```
dt.predict([[5,6]])
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/64faf67d-cafa-4d54-a7c6-9a9e2cdd97e4)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
