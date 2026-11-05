# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: T.Rishabh srivatsav
RegisterNumber: 212224113001
*/

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
df = pd.read_csv("student_scores.csv")
df.head(10)
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x = df.iloc[:,0:1]
y = df.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['Hours'],df["Scores"])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train,lr.predict(X_train),color='red')
lr.coef_
lr.intercept_
y_pred = lr.predict(X_test)
mse = mean_squared_error(Y_test,y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test,y_pred)
r2 = r2_score(Y_test,y_pred)
print("MSE:",mse)
print("RMSE:",rmse)
print("MAE:",mae)
print("R2:",r2)

```

## Output:

<img width="1271" height="725" alt="Screenshot 2026-02-02 140437" src="https://github.com/user-attachments/assets/12ae7b43-9b77-4295-9841-aeb0a6da7e7d" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
