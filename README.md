# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

  1.Startv the program.
  
  2.Import numpy as np.
  
  3.Give the header to the data.
  
  4.Find the profit of population.
  
  5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
  
  6.End the program.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Mohamed Athif Rahuman J
RegisterNumber:212223220058
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    """
    Take ina numpy array X,y,theta and generate the cost function of using the in a linear regression model
    """
    m=len(y) #length of the training data
    h=X.dot(theta)#hypothesis
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) #returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
      predictions=X.dot(theta)
      error=np.dot(X.transpose(),(predictions-y))
      descent=alpha*1/m*error
      theta-=descent
      J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70000,we predict a profit of $"+str(round(predict2,0)))
```

## Output:

## PROFIT PREDICTION GRAPH:

![Screenshot 2023-09-17 085931](https://github.com/RENUGASARAVANAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119292258/c99c236f-fbd2-4b93-b94d-88b2bae00e67)

## COMPUTE COST VALUE:

![Screenshot 2023-09-17 090047](https://github.com/RENUGASARAVANAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119292258/8ac9bf73-a895-4de9-88ce-b579fe141387)

## h(x) VALUE:

![Screenshot 2023-09-17 090059](https://github.com/RENUGASARAVANAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119292258/8da5cc0c-9dc2-42a1-baae-175734b8cae7)

## COST FUNCTION USING GRADIENT DESCENT GRAPH:

![Screenshot 2023-09-17 090147](https://github.com/RENUGASARAVANAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119292258/1b30479a-e925-4dbf-ba31-385d3821b8c0)

## PROFIT PREDICTION GRAPH:

![Screenshot 2023-09-17 090200](https://github.com/RENUGASARAVANAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119292258/2a1fe829-35cf-4a3c-89d5-7cbb307dea5c)

## PROFIT FOR THE POPULATION 35,000:

![Screenshot 2023-09-17 090252](https://github.com/RENUGASARAVANAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119292258/2c26ebc8-59d7-43dc-bd4a-68ef58996b35)

## PROFIT FOR THE POPULATION 70,000:

![Screenshot 2023-09-17 090259](https://github.com/RENUGASARAVANAN/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119292258/30fd7a04-a6d2-4a03-a17f-53340d4f6363)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
