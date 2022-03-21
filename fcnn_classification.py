
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from random import shuffle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


path_c11 = 'Dataset/Classification/LS_Group26/Class1.txt'
path_c12 = 'Dataset/Classification/LS_Group26/Class2.txt'
path_c13 = 'Dataset/Classification/LS_Group26/Class3.txt'
path_c2 = 'Dataset/Classification/NLS_Group26.txt'


def recover(df):
    s1 = []
    s2 = []
    for index, row in df.iterrows():
        val = row[0]
        val = val[:-1]
        val = list(map(float, val.split()))
        s1.append(val[0])
        s2.append(val[1])
    df = pd.DataFrame(list(zip(s1, s2)))
    return df


def split_c(frames):
    y_true = []
    for i in range(len(frames)):
        for j in range(len(frames[i])):
            y_true.append(i+1)
    df = pd.concat(frames)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y_true, test_size=0.3, shuffle=True, random_state=59)
    return X_train, X_test, y_train, y_test


def train_test_valid_split(frames):
    y_true = []
    for i in range(len(frames)):
        for j in range(len(frames[i])):
            y_true.append(i+1)
    df = pd.concat(frames)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y_true, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


def instantaneous_error(y, s):
    E = 0
    for i in range(len(y)):
        E += (0.5*(y[i]-s[i])**2)
    return E


def logistics_fn(a):
    return 1/(1+np.exp(-a))


def logisticsAr_fn(a):
    ls = []
    for i in range(len(a)):
        ls.append(logistics_fn(a[i]))
    return ls


def derivative_logistics(a):
    return logistics_fn(a)*(1-logistics_fn(a))


def derivative_logistics_arr(a):
    ls = []
    for i in range(len(a)):
        ls.append(derivative_logistics(a[i]))
    return ls


def generate_wt(x, y):
    l = []
    for i in range(x * y):
        l.append(np.random.randn())
    return(np.array(l).reshape(x, y))


def loss(out, Y):
    s = (np.square(np.array(out)-np.array(Y)))
    s = np.sum(s)/len(y)
    return(s)


def forward_computation(x, w):
    a = []
    df = []
    z = x.dot(np.array(w[0]))
    s = logisticsAr_fn(z)
    df.append(derivative_logistics_arr(z))
    a.append(s)
    for i in range(1, len(w)):
        z = np.array(s).dot(np.array(w[i]))
        s = logisticsAr_fn(z)
        df.append(derivative_logistics_arr(z))
        a.append(s)
    return s, a, df


def back_prop(x, y, w, alpha, a, df):
    d = []
    a = np.array(a)
    x = x.reshape((1, len(x)))
    d.append(np.array(a[-1])-np.array(y))
    i = len(w)-2
    while(i >= 0):
        d.append(np.multiply((np.array(w[i+1]).dot((np.array(d[-1]).transpose()))).transpose(),
                             (np.multiply(np.array(a[i]), 1-np.array(a[i])))))
        i -= 1
    d = d[::-1]
    w_adj = (x.transpose().dot(np.array(d[0]).reshape((1, -1))))
    w[0] = w[0]-(alpha*w_adj)
    for i in range(1, len(w)):
        w_adj = np.array(a[i-1]).transpose().reshape((-1, 1)
                                                     ).dot(np.array(d[i]).reshape((1, -1)))
        w[i] = w[i]-(alpha*w_adj)
    return w


def train(X, y, w, eta=0.01, epoch=100, task="classification"):
    acc = []
    losss = []
    Error_matrix = []
    for j in range(epoch):
        l = []
        Eav = 0
        for i in range(len(X)):
            s, a, df = forward_computation(X[i], w)
            Eav += instantaneous_error(s, y[i])
            w = back_prop(X[i], y[i], w, eta, a, df)
            l.append((loss(s, y[i])))
        Eav /= len(X)
        Error_matrix.append(Eav)
        acc.append((1-(sum(l)/len(X)))*100)
        losss.append(sum(l)/len(X))
    return acc, losss, w, Error_matrix


def predict(test, W):
    y_pred = []
    test = np.array(test)
    for i in range(len(test)):
        pred, a, df = forward_computation(test[i], W)
        if(max(pred)) == pred[0]:
            y_pred.append(1)
        elif(max(pred) == pred[1]):
            y_pred.append(2)
        else:
            y_pred.append(3)
    return y_pred


df_c11 = pd.read_csv(path_c11, sep=" ", header=None)
df_c12 = pd.read_csv(path_c12, sep=" ", header=None)
df_c13 = pd.read_csv(path_c13, sep=" ", header=None)

X_train, y_train, X_val, y_val, X_test, y_test = train_test_valid_split(
    [df_c11, df_c12, df_c13])

param_set = []
for i in range(4, 10, 2):
    for j in range(3, 9, 2):
        param_set.append([2, i, j, 3])

W_param = []
Error_matrix = []
for neuron_count in param_set:
    w = []
    for i in range(1, len(neuron_count)):
        arr = np.random.random((neuron_count[i-1], neuron_count[i]))
        arr = list(arr)
        w.append(arr)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y = []
    for i in range(len(y_train)):
        if(y_train[i] == 1):
            y.append([1, 0, 0])
        elif(y_train[i] == 2):
            y.append([0, 1, 0])
        else:
            y.append([0, 0, 1])
    a, l, w, Error = train(X_train, y, w)
    y_pred = predict(X_val, w)
    Error_matrix.append(Error)
    W_param.append(w)
    print("Neural netwerk (nodes in each layer )", neuron_count)
    print("---->Accuracy :", accuracy_score(y_pred, y_val))
    print("---->Confusion matrix : \n", confusion_matrix(y_val, y_pred))
    print()

Error = Error_matrix[0]
w = W_param[0]

plt.plot(Error)
plt.xlabel("no. of epochs")
plt.ylabel("Average error")
title_ = "Average error vs no. of epochs"
plt.title(title_)
plt.show()

print("For training data")
y_pred = predict(X_train, w)
print("Accuracy : ", accuracy_score(y_train, y_pred))
print("Confusion martix : \n", confusion_matrix(y_train, y_pred))

print("For validation data")
y_pred = predict(X_val, w)
print("Accuracy : ", accuracy_score(y_val, y_pred))
print("Confusion martix : \n", confusion_matrix(y_val, y_pred))

print("For test data")
y_pred = predict(X_test, w)
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Confusion martix : \n", confusion_matrix(y_test, y_pred))


def plot_surface(X_train, w):
    steps = 100
    X_train = np.array(X_train)
    x_span = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), steps)
    y_span = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), steps)
    xx, yy = np.meshgrid(x_span, y_span)
    arr = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for i in range(len(arr)):
        x = [arr[i]]
        pred = predict(x, w)
        Z.append(pred[0])
    Z = np.array(Z).reshape(xx.shape)
    y_pred = predict(X_train, w)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    sns.scatterplot(X_train[:, 0], X_train[:, 1], hue=y_pred)
    plt.title("Decision boundary class-1,2,3")
    plt.show()


plot_surface(X_train, w)


def plot_all(X, w):
    output = []
    for i in range(len(X)):
        s, a, df = forward_computation(X[i], w)
        temp = []
        for j in range(len(a)):
            for k in range(len(a[j])):
                temp.append(a[j][k])
        output.append(temp)
    for i in range(len(output[0])):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.array(X)[:, 0], np.array(X)[
                   :, 1], np.array(output)[:, i],)
        plt.show()


plot_all(X_train, w)

plot_all(np.array(X_val), w)

plot_all(np.array(X_test), w)

df_c21 = pd.read_csv(path_c2, skiprows=1, nrows=500, header=None)
df_c22 = pd.read_csv(path_c2, skiprows=501, nrows=500, header=None)
df_c23 = pd.read_csv(path_c2, skiprows=1001, nrows=1000, header=None)
df_c21 = recover(df_c21)
df_c22 = recover(df_c22)
df_c23 = recover(df_c23)

X_train, y_train, X_val, y_val, X_test, y_test = train_test_valid_split(
    [df_c21, df_c22, df_c23])

param_set = []
for i in range(4, 10, 2):
    for j in range(3, 9, 2):
        param_set.append([2, i, j, 3])

W_param = []
Error_matrix = []
for neuron_count in param_set:
    w = []
    for i in range(1, len(neuron_count)):
        arr = np.random.random((neuron_count[i-1], neuron_count[i]))
        arr = list(arr)
        w.append(arr)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y = []
    for i in range(len(y_train)):
        if(y_train[i] == 1):
            y.append([1, 0, 0])
        elif(y_train[i] == 2):
            y.append([0, 1, 0])
        else:
            y.append([0, 0, 1])
    a, l, w, Error = train(X_train, y, w)
    y_pred = predict(X_val, w)
    Error_matrix.append(Error)
    W_param.append(w)
    print("Neural netwerk (nodes in each layer )", neuron_count)
    print("---->Accuracy :", accuracy_score(y_pred, y_val))
    print("---->Confusion matrix : \n", confusion_matrix(y_val, y_pred))
    print()

Error = Error_matrix[5]
w = W_param[5]

plt.plot(Error)
plt.xlabel("no. of epochs")
plt.ylabel("Average error")
title_ = "Average error vs no. of epochs"
plt.title(title_)
plt.show()

y_pred = predict(X_train, w)
print("For training data")
print("Accuracy : ", accuracy_score(y_train, y_pred))
print("Confusion martix : \n", confusion_matrix(y_train, y_pred))

y_pred = predict(X_val, w)
print("For validation data")
print("Accuracy : ", accuracy_score(y_val, y_pred))
print("Confusion martix : \n", confusion_matrix(y_val, y_pred))

y_pred = predict(X_test, w)
print("For test data")
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Confusion martix : \n", confusion_matrix(y_test, y_pred))

plot_surface(X_train, w)

plot_all(X_train, w)

plot_all(np.array(X_val), w)

plot_all(np.array(X_test), w)
