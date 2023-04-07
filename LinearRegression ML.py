import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data=pd.read_csv("placement.csv")
X=data['cgpa']
Y=data['placement_exam_marks']
xval=X.values
yval=Y.values
x=xval.reshape(-1,1)
y=yval.reshape(-1,1)

#splitting
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=21)

#training
lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)

#predicting
test_pred=lin_reg.predict(X_test)

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'black')
plt.plot(X_train, lin_reg.predict(X_train), color = 'red')
plt.title('placements(Test set)')
plt.xlabel('cgpa')
plt.ylabel('placement_exam_marks')
plt.show()

#Evaluating the Model
score=r2_score(Y_test,test_pred)
print("R2 Score is =",score) #printing the accuracy
print("MSE is =",mean_squared_error(Y_test,test_pred))
print("RMSE of is =",np.sqrt(mean_squared_error(Y_test,test_pred)))

import pickle
model = pickle.dump(lin_reg, open('model10.pkl','wb'))


