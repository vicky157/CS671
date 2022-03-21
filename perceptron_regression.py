
#---------Libraries----------------#
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from random import shuffle
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error


#----path to input files-----------------#
path_r1 = 'Dataset/Regression/UnivariateData/26.csv'
path_r2 = 'Dataset/Regression/BivariateData/26.csv'

#----------Function to split data into train,test----------#
def split_c(frames):
  y_true=[]
  for i in range(len(frames)):
    for j in range(len(frames[i])):
      y_true.append(i+1)
  df=pd.concat(frames)
  X_train,X_test,y_train,y_test=train_test_split(df,y_true,test_size=0.3,shuffle=True,random_state=59)
  return X_train,X_test,y_train,y_test

#---------Function to split data into train,test, validation set-----#
def train_test_valid_split(frames):
  y_true=[]
  for i in range(len(frames)):
    for j in range(len(frames[i])):
      y_true.append(i+1)
  df=pd.concat(frames)
  X_train,X_test,y_train,y_test=train_test_split(df,y_true,test_size=0.2,random_state=42)
  X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.25,random_state=42)
  return X_train,y_train,X_val,y_val,X_test,y_test

def instantaneous_error(y,s):
  return 0.5*(y-s)**2

def logistics_fn(a):
  return 1/(1+np.exp(-a))

def logisticsAr_fn(a):
  ls=[]
  for i in range(len(a)):
    ls.append(logistics_fn(a[i]))
  return ls

def derivative_logistics(a):
  return logistics_fn(a)*(1-logistics_fn(a))

def derivative_logistics_arr(a):
  ls=[]
  for i in range(len(a)):
    ls.append(derivative_logistics(a[i]))
  return ls

#--------------function impmlements a perceptron--------------#
def perceptron(X,y,eta,max_epoch,task,convergence_threshold=0.0000001):
  X=np.array(X)
  y=np.array(y)
  n=len(X)
  d=len(X[0])
  w=np.random.rand(d+1) #weight vector intitiation
  max_epoch=25
  epoch=0
  Error_matrix=[]
  while(True):
    epoch+=1
    Eav=0
    for i in range(n):
      a=np.dot(np.array((w)[:-1]),X[i])+w[-1]
      s=logistics_fn(a)
      if(task=="regression"):
        s=a
      E=instantaneous_error(y[i],s)
      Eav+=E
      if(task=="classification"):
        dw=eta*(y[i]-s)*derivative_logistics(a)*X[i]
        dw=list(dw)
        dw.append(eta*(y[i]-s)*derivative_logistics(a))
        dw=np.array(dw)
      else: #calculate change in weight vector
        dw=eta*(y[i]-s)*X[i]
        dw=list(dw)
        dw.append(eta*(y[i]-s))
        dw=np.array(dw)        
      w=np.add(np.array(w),np.array(dw))#--update weights-----#
    Eav/=n
    Error_matrix.append(Eav)
    if( epoch>max_epoch):
      return w,Error_matrix

"""**Classifier using perceptron**"""

#----------function to perform prediction using perceptron------#
def predict_perceptron_regressor(w,test):
  test=np.array(test)
  y_pred=[]
  for i in range(len(test)):
    y_pred.append(np.dot(np.array((w)[:-1]),test[i])+w[-1])
  return y_pred

#---function to train model --------#
def train(X_train,y_train,eta,max_epoch):
  X_train=np.array(X_train)
  w,Error=perceptron(X_train,y_train,eta,max_epoch,"regression")
  plt.plot(Error)
  plt.xlabel("no. of epochs")
  plt.ylabel("Average error")
  title_="Average error vs no. of epochs"
  plt.title(title_)
  plt.show()
  return w

df_r1=pd.read_csv(path_r1,header=None)
y=df_r1[1]
df_r1 = df_r1.drop(1, 1)

X_train,X_test,y_train,y_test=train_test_split(df_r1,y,test_size=0.3,shuffle=True,random_state=42)

sns.scatterplot(np.array(X_train)[:,0],np.array(y_train))
plt.title("True plot of training data")
plt.show()

#call function to train model
eta=0.05
max_epoch=100
w=train(X_train,y_train,eta,max_epoch)

y_pred=predict_perceptron_regressor(w,X_train)

train_mse=mean_squared_error(y_pred, y_train)

y_pred=predict_perceptron_regressor(w,np.array(X_test))

test_mse=mean_squared_error(y_pred,y_test)

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
plt.bar(["train","test"],[train_mse,test_mse])
addlabels(["train","test"],[train_mse,test_mse])
plt.title("MSE plot")
plt.show()

y_pred=predict_perceptron_regressor(w,X_train)

plt.scatter(X_train,y_train)
plt.scatter(X_train,y_pred)
plt.title("Model output, target output, training data")
plt.show()
plt.scatter(y_train,y_pred)
plt.title("Model output vs target output, training data")
plt.xlabel("target output")
plt.ylabel("Model output")
plt.show()

y_pred=predict_perceptron_regressor(w,X_test)

plt.scatter(X_test,y_test)
plt.scatter(X_test,y_pred)
plt.title("Model output, target output, test data")
plt.show()
plt.scatter(y_test,y_pred)
plt.title("Model output vs target output, test data")
plt.xlabel("target output")
plt.ylabel("Model output")
plt.show()

df_r2=pd.read_csv(path_r2,header=None)
y=df_r2[2]
df_r2 = df_r2.drop(2, 1)

X_train,X_test,y_train,y_test=train_test_split(df_r2,y,test_size=0.3,shuffle=True,random_state=42)

fig = plt.figure()    
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(np.array(X_train)[:,0], np.array(X_train)[:,1], y_train,s=0.5) 
plt.title("True plot of training data")
plt.show()

eta=0.05
max_epoch=100
w=train(X_train,y_train,eta,max_epoch)

y_pred=predict_perceptron_regressor(w,X_train)

train_mse=mean_squared_error(y_pred, y_train)

y_pred=predict_perceptron_regressor(w,np.array(X_test))

test_mse=mean_squared_error(y_pred,y_test)

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
plt.bar(["train","test"],[train_mse,test_mse])
addlabels(["train","test"],[train_mse,test_mse])
plt.title("MSE plot")
plt.show()

y_pred=predict_perceptron_regressor(w,X_train)

fig = plt.figure()    
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(np.array(X_train)[:,0], np.array(X_train)[:,1], y_train,s=0.5) 
ax.scatter(np.array(X_train)[:,0], np.array(X_train)[:,1], y_pred,s=0.5) 
plt.title("given curved surface and predicted plane, training data")
plt.show()

y_pred=predict_perceptron_regressor(w,X_test)

fig = plt.figure()    
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(np.array(X_test)[:,0], np.array(X_test)[:,1], y_test,s=0.5) 
ax.scatter(np.array(X_test)[:,0], np.array(X_test)[:,1], y_pred,s=0.5) 
plt.title("given curved surface and predicted plane, test data")
plt.show()

y_pred=predict_perceptron_regressor(w,X_train)
plt.scatter(y_train,y_pred)
plt.title("Model output vs target output, training data")
plt.xlabel("target output")
plt.ylabel("Model output")
plt.show()

y_pred=predict_perceptron_regressor(w,X_test)
plt.scatter(y_test,y_pred)
plt.title("Model output vs target output, test data")
plt.xlabel("target output")
plt.ylabel("Model output")
plt.show()
