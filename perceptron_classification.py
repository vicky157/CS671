
#---------Libraries----------------#
import numpy as np
import pandas as pd
from sklearn import svm
import seaborn as sns
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split



#----path to input files-----------------#
path_c11 = 'Dataset/Classification/LS_Group26/Class1.txt'
path_c12 = 'Dataset/Classification/LS_Group26/Class2.txt'
path_c13 = 'Dataset/Classification/LS_Group26/Class3.txt'
path_c2 = 'Dataset/Classification/NLS_Group26.txt'

#--------function to retrieve input from txt file in desired format----------#
def recover(df):
  s1=[]
  s2=[]
  for index,row in df.iterrows():
    val=row[0]
    val=val[:-1]
    val=list(map(float, val.split()))
    s1.append(val[0])
    s2.append(val[1])
  df = pd.DataFrame(list(zip(s1, s2)))
  return df

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
  w=np.random.rand(d+1)  #weight vector intitiation 
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
      if(task=="classification"):   #calculate change in weight vector
        dw=eta*(y[i]-s)*derivative_logistics(a)*X[i]
        dw=list(dw)
        dw.append(eta*(y[i]-s)*derivative_logistics(a))
        dw=np.array(dw)
      else:
        dw=eta*(y[i]-s)*X[i]
        dw=list(dw)
        dw.append(eta*(y[i]-s))
        dw=np.array(dw)

      w=np.add(np.array(w),np.array(dw)) #--update weights-----#
    Eav/=(n)
    Error_matrix.append(Eav)
    if( epoch>max_epoch):
      return w,Error_matrix

"""**Classifier using perceptron**"""

#----------function to perform prediction using perceptron------#
def predict_perceptron_classifier(w,test):
  y_pred=[]
  for k in range(len(test)):
    count_class=[0]*3
    count=0
    for i in range(2): #-----classifies using one against one approach and decides final label on basis of majority
      for j in range(i+1,3):
        a=np.dot(np.array(w[count][:-1]),test[k])+w[count][-1]
        s=logistics_fn(a)
        if(s>0.5):
          count_class[j]+=1
        else:
          count_class[i]+=1
        count+=1
    if(count_class[0]==max(count_class)):
        y_pred.append(1)
    elif(count_class[1]==max(count_class)):
         y_pred.append(2)
    else:
        y_pred.append(3) 
  return y_pred

#----function to plot decision surface
def plot_surface(X_train,w,c1,c2):
  steps = 100
  X_train=np.array(X_train)
  x_span = np.linspace(X_train[:,0].min(), X_train[:,0].max(), steps)
  y_span = np.linspace(X_train[:,1].min(), X_train[:,1].max(), steps)
  xx, yy = np.meshgrid(x_span, y_span)

  arr=np.c_[xx.ravel(), yy.ravel()]
  Z=[]
  for i in range(len(arr)):
    a=np.dot(np.array(w[:-1]),arr[i])+w[-1]
    s=logistics_fn(a)
    if(s>0.5):
      Z.append(c2)
    else:
      Z.append(c1)
  Z=np.array(Z).reshape(xx.shape)
  y_pred=[]
  for i in range(len(X_train)):
    a=np.dot(np.array(w[:-1]),X_train[i])+w[-1]
    s=logistics_fn(a)
    if(s>0.5):
      y_pred.append(c2)
    else:
      y_pred.append(c1)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  title_="Decision boundary class "+str(c1)+" vs "+str(c2)
  plt.title(title_)
  sns.scatterplot(X_train[:,0], X_train[:,1],hue=y_pred)
  plt.show()

def plot_all(X_train,w):
  steps = 100
  X_train=np.array(X_train)
  x_span = np.linspace(X_train[:,0].min(), X_train[:,0].max(), steps)
  y_span = np.linspace(X_train[:,1].min(), X_train[:,1].max(), steps)
  xx, yy = np.meshgrid(x_span, y_span)
  arr=np.c_[xx.ravel(), yy.ravel()]
  Z=[]
  for i in range(len(arr)):
    x=[arr[i]]
    pred=predict_perceptron_classifier(w,x)
    Z.append(pred[0])
  Z=np.array(Z).reshape(xx.shape)
  y_pred=predict_perceptron_classifier(w,X_train)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  sns.scatterplot(X_train[:,0], X_train[:,1],hue=y_pred)
  plt.title("Decision boundary class-1,2,3")
  plt.show()

#---function to train model --------#
def train(X_train,y_train,eta,max_epoch):
  X_train=np.array(X_train)
  w=[]
  n=len(y_train)
  for i in range(2): #uses one against one approach
    for j in range(i+1,3):
      y_=[]
      X=[]
      for k in range(n):
        if(y_train[k]==i+1):
          y_.append(0)
          X.append(X_train[k])
        elif(y_train[k]==j+1):
          y_.append(1)
          X.append(X_train[k])
      param,Error=perceptron(X,y_,eta,max_epoch,"classification")
      plot_surface(X,param,i+1,j+1)
      w.append(param)
      plt.plot(Error)
      plt.xlabel("no. of epochs")
      plt.ylabel("Average error")
      title_="Average error vs no. of epochs, class-"+str(i+1)+" vs "+str(j+1)
      plt.title(title_)
      plt.show()
  return w

df_c11=pd.read_csv(path_c11, sep=" ", header=None)
df_c12=pd.read_csv(path_c12, sep=" ", header=None)
df_c13=pd.read_csv(path_c13, sep=" ", header=None)

frames=[df_c11,df_c12,df_c13]
X_train,X_test,y_train,y_test=split_c(frames)
sns.scatterplot(np.array(X_train)[:,0],np.array(X_train)[:,1],hue=y_train)
plt.title("True plot of training data")
plt.show()

#call function to train model
eta=0.05
max_epoch=100
w=train(X_train,y_train,eta,max_epoch)

plot_all(X_train,w)

y_pred=predict_perceptron_classifier(w,np.array(X_train))
print("For training data")
print("Accuracy : ",accuracy_score(y_pred,y_train))
print("Confusion martix : \n",confusion_matrix(y_pred,y_train))

y_pred=predict_perceptron_classifier(w,np.array(X_test))
print("For test data")
print("Accuracy : ",accuracy_score(y_pred,y_test))
print("Confusion martix : \n",confusion_matrix(y_pred,y_test))

#read input file 2
df_c21=pd.read_csv(path_c2, skiprows = 1, nrows=500,header=None)
df_c22=pd.read_csv(path_c2, skiprows = 501, nrows=500,header=None)
df_c23=pd.read_csv(path_c2, skiprows = 1001, nrows=1000,header=None)
df_c21=recover(df_c21)
df_c22=recover(df_c22)
df_c23=recover(df_c23)

frames=[df_c21,df_c22,df_c23]
X_train,X_test,y_train,y_test=split_c(frames)
sns.scatterplot(np.array(X_train)[:,0],np.array(X_train)[:,1],hue=y_train)
plt.title("True plot of training data")
plt.show()

#call function to train model
eta=0.05
max_epoch=100
w=train(X_train,y_train,eta,max_epoch)

plot_all(X_train,w)

y_pred=predict_perceptron_classifier(w,np.array(X_train))
print("For training data")
print("Accuracy : ",accuracy_score(y_pred,y_train))
print("Confusion martix : \n",confusion_matrix(y_pred,y_train))

y_pred=predict_perceptron_classifier(w,np.array(X_test))
print("For test data")
print("Accuracy : ",accuracy_score(y_pred,y_test))
print("Confusion martix : \n",confusion_matrix(y_pred,y_test))
