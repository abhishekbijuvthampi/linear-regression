import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")
split = int(len(df)*0.8)

df_train = df[:split]
df_test  = df[split:]

x_train = df_train['x']
y_train = df_train['y']

x_test = df_test['x']
y_test = df_test['y']

def error(x, y, w, b):
    m = x.shape[0]
    sum = 0
    for i in range(m):
        sum += (((w*x[i])+b) - y[i])**2
        return sum/(2*m)
    
def grads(x, y, w, b):
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w*x[i] + b

        dj_db += f_wb - y[i]
        dj_dw += (f_wb - y[i])*x[i]

        return dj_db/m, dj_dw/m
    
def grad_des(x, y, w, b, n_iter, learn_rate):
    error_his = []
    wb_his = []
    
    new_w = w
    new_b = b

    for i in range(n_iter):

        dj_db, dj_dw = grads(x, y, new_w, new_b)

        new_w = new_w - learn_rate*dj_dw
        new_b = new_b - learn_rate*dj_db

        error_his.append(error(x, y, new_w, new_b))
        wb_his.append([new_w, new_b])

    return new_w, new_b, error_his, wb_his

first_w = 5
first_b = -1

iter = 10000
rate = 0.01

final_w, final_b, error_his, wb_his = grad_des(x_train, y_train, first_w, first_b, iter, rate)

final_wb = final_w*x_train - final_b


def perdict(test):
    
    pred_f = (final_w * test) + final_b

    return(pred_f)

x_train = np.array(x_train)
y_train = np.array(y_train)

#trained
fig, ax = plt.subplots(1, 2)
ax[0].scatter(x_train, y_train, marker='x', c='r')
ax[0].plot(x_train, final_wb)
ax[0].set_title("Trained")

#perdicted
ax[1].scatter(x_test, perdict(x_test), marker='x', c='b')
ax[1].set_title("Predicted")
plt.show()
