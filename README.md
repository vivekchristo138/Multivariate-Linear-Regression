# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br>

### Step2
<br>

### Step3
<br>

### Step4
<br>

### Step5
<br>

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
boston=datasets.load_diabetes(return_X_y=False)

#defining feature matrix(X) and response vector (y)
x=boston.data
y=boston.target
#splitting x and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

#create linear regression object
reg=linear_model.LinearRegression()

#train the model using the training sets
reg.fit(x_train,y_train)

#regression coefficients
print("Coefficients",reg.coef_)

#variance score: 1means perfect prediction
print("Variance score: {}".format(reg.score(x_test,y_test)))

#plot for residual error
#setting plot style
plt.style.use("fivethirtyeight")

#plotting residual errors in training data
plt.scatter(reg.predict(x_train),reg.predict(x_train)-y_train,color='green',s=10,label="Train data")

#plotting residual errors in test data
plt.scatter(reg.predict(x_test),reg.predict(x_test)-y_test,color='blue',s=10,label="Test data")

#plotting line for zero residual error
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)

#plotting legend
plt.legend(loc='upper right')

#plot title
plt.title('Residual errors')

##method call for showing the plot
plt.show()





```
## Output:

<img width="910" height="738" alt="image" src="https://github.com/user-attachments/assets/e997122e-ea13-4f9b-86b8-db79b2a99274" />


<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
