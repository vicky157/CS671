
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


path_r1 = 'Dataset/Regression/UnivariateData/26.csv'
path_r2 = 'Dataset/Regression/BivariateData/26.csv'


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


def rmse(a, b):
    mse = 0
    for i in range(len(a)):
        mse += (a[i]-b[i])**2
    mse /= (len(a))
    mse **= 0.5
    return mse


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
    return 0.5*(y-s)**2


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


def forward_computation(x, w, task):
    a = []
    df = []
    z = x.dot(np.array(w[0]))
    s = logisticsAr_fn(z)
    df.append(derivative_logistics_arr(z))
    a.append(s)
    for i in range(1, len(w)):
        z = np.array(s).dot(np.array(w[i]))
        if(i == len(w)-1 and task == "regression"):
            s = z
            a.append(s)
            continue
        s = logisticsAr_fn(z)
        df.append(derivative_logistics_arr(z))
        a.append(s)
    return s, a, df


def back_prop(x, y, w, alpha, a, df, task):
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


def train(X, y, w, eta=0.05, epoch=100, task="regression"):
    acc = []
    losss = []
    Error_matrix = []
    for j in range(epoch):
        l = []
        Eav = 0
        for i in range(max(5, len(X))):
            s, a, df = forward_computation(X[i], w, task)
            Eav += instantaneous_error(s, y[i])
            w = back_prop(X[i], y[i], w, eta, a, df, task)
            l.append((loss(s, y[i])))
        Eav /= len(X)
        Error_matrix.append(Eav)
        acc.append((1-(sum(l)/len(X)))*100)
        losss.append(sum(l)/len(X))
    return acc, losss, w, Error_matrix


def predict(test, W, task="regression"):
    y_pred = []
    test = np.array(test)
    for i in range(len(test)):
        pred, a, df = forward_computation(test[i], W, task)
        if(task == "regression"):
            y_pred.append(pred)
            continue
        if(max(pred)) == pred[0]:
            y_pred.append(1)
        elif(max(pred) == pred[1]):
            y_pred.append(2)
        else:
            y_pred.append(3)
    return y_pred


df_r1 = pd.read_csv(path_r1, header=None)
y = df_r1[1]
df_r1 = df_r1.drop(1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    df_r1, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42)

param_set = []
for i in range(4, 10, 2):
    for j in range(3, 9, 2):
        param_set.append([1, i, j, 1])

W_param = []
Error_matrix = []
for neuron_count in param_set:
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    w = []
    for i in range(1, len(neuron_count)):
        arr = np.random.random((neuron_count[i-1], neuron_count[i]))
        arr = list(arr)
        w.append(arr)
    y = []
    a, l, w, error = train(X_train, y_train, w, 0.05, 100, "regression")
    Error_matrix.append(error)
    y_pred = predict(X_val, w, "regression")
    W_param.append(w)
    print("Neural netwerk (nodes in each layer )", neuron_count)
    print("----->Root mean square error : ", rmse(y_pred, np.array(y_val)))
    print()

Error = Error_matrix[8]
w = W_param[8]

plt.plot(Error)
plt.xlabel("no. of epochs")
plt.ylabel("Average error")
title_ = "Average error vs no. of epochs"
plt.title(title_)
plt.show()

y_pred = predict(X_train, w, "regression")
print("RMSE error for train set", rmse(y_train, y_pred))

y_pred = predict(np.array(X_val), w, "regression")
print("RMSE error for validation set", rmse(np.array(y_val), y_pred))

y_pred = predict(np.array(X_test), w, "regression")
print("RMSE error for test set", rmse(np.array(y_test), y_pred))


def compare_actual(X, y_true, w, title_):
    y_pred = predict(X, w, "regression")
    plt.scatter(X, y_true)
    plt.scatter(X, y_pred)
    plt.title("Model output, target output,"+title_+" data")
    plt.show()
    plt.scatter(y_true, y_pred)
    plt.title("Model output vs target output,"+title_+" data")
    plt.xlabel("target output")
    plt.ylabel("Model output")
    plt.show()


compare_actual(X_train, y_train, w, "training")

compare_actual(np.array(X_val), y_val, w, "validation")

compare_actual(np.array(X_test), y_test, w, "test")


def plot_all(X, w):
    output = []
    for i in range(len(X)):
        s, a, df = forward_computation(X[i], w, 'regression')
        temp = []
        for j in range(len(a)):
            for k in range(len(a[j])):
                temp.append(a[j][k])
        output.append(temp)
    for i in range(len(output[0])):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.array(X)[:, 0], np.array(output)[:, i],)
        plt.show()


plot_all(X_train, w)

plot_all(np.array(X_val), w)

plot_all(np.array(X_test), w)

df_r2 = pd.read_csv(path_r2, header=None)
y = df_r2[2]
df_r2 = df_r2.drop(2, 1)

X_train, X_test, y_train, y_test = train_test_split(
    df_r2, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42)

param_set = []
for i in range(4, 10, 2):
    for j in range(3, 9, 2):
        param_set.append([2, i, j, 1])

W_param = []
Error_matrix = []
for neuron_count in param_set:
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    w = []
    for i in range(1, len(neuron_count)):
        arr = np.random.random((neuron_count[i-1], neuron_count[i]))
        arr = list(arr)
        w.append(arr)
    y = []
    a, l, w, error = train(X_train, y_train, w, 0.05, 100, "regression")
    Error_matrix.append(error)
    y_pred = predict(X_val, w, "regression")
    W_param.append(w)
    print("Neural netwerk (nodes in each layer )", neuron_count)
    print("----->Root mean square error : ", rmse(y_pred, np.array(y_val)))
    print()

w = W_param[7]
Error = Error_matrix[7]

plt.plot(Error)
plt.xlabel("no. of epochs")
plt.ylabel("Average error")
title_ = "Average error vs no. of epochs"
plt.title(title_)
plt.show()

y_pred = predict(X_train, w, "regression")
print("RMSE error for train set", rmse(y_train, y_pred))

y_pred = predict(np.array(X_val), w, "regression")
print("RMSE error for validatoin set", rmse(np.array(y_val), y_pred))

y_pred = predict(np.array(X_test), w, "regression")
print("RMSE error for test set", rmse(np.array(y_test), y_pred))


def compare_actual_3d(X, y_true, w, title_):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], y_true, s=0.5)
    y_pred = predict(X, w, "regression")
    ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], y_pred, s=0.5)
    plt.title(title_)
    plt.show()
    plt.scatter(y_true, y_pred)
    plt.title(title_)
    plt.xlabel("target output")
    plt.ylabel("Model output")
    plt.show()


compare_actual_3d(X_train, y_train, w,
                  "Model output and target output, training data")

compare_actual_3d(np.array(X_val), y_val, w,
                  "Model output and target output, validation data")

compare_actual_3d(np.array(X_test), y_test, w,
                  "Model output and target output, test data")


def plot_all(X, w):
    output = []
    for i in range(len(X)):
        s, a, df = forward_computation(X[i], w, "regression")
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
