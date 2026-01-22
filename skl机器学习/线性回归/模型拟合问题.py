import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
import matplotlib.pyplot as plt

def dm_模型拟合_欠拟合():
    # 设定随机种子,每次生成随机数一致
    np.random.seed(42)
    # 设置x值生成-3到3之间的随机正态分布
    x = np.random.uniform(-3,3,size=100)
    # 设置y的真实值,关于x的2次函数加上0-1之间的随机数值(噪声)
    y = 0.5*x**2+x+10+np.random.normal(0,1,size=100)
    # 塑性
    x = x.reshape(-1,1)
    # 模型训练
    model = LinearRegression()
    model.fit(x,y)
    # 预测
    y_p = model.predict(x)
    # 评分
    print(root_mean_squared_error(y_p, y))
    print(mean_squared_error(y_p, y))
    print(mean_absolute_error(y_p, y))

    plt.scatter(x, y)
    plt.plot(x, y_p,color='red')
    plt.show()

def dm_模型拟合_拟合():

    np.random.seed(18)
    x = np.random.uniform(-3,3,size=100)
    x1 = x.reshape(-1, 1)
    y = 0.5*x**2+x+10+np.random.normal(0,1,size=100)
    # 给x多加一列特征来增强拟合效果
    x2 = np.hstack([x1, x1 ** 2])

    model = LinearRegression()
    model.fit(x2,y)
    y_p = model.predict(x2)
    print(root_mean_squared_error(y_p, y))
    print(mean_squared_error(y_p, y))
    print(mean_absolute_error(y_p, y))

    plt.scatter(x, y)
    plt.plot(x[np.argsort(x)], y_p[np.argsort(x)],color='red')
    plt.show()

def dm_模型拟合_过拟合():
    np.random.seed(18)
    x = np.random.uniform(-3,3,size=100)
    x1 = x.reshape(-1, 1)
    y = 0.5*x**2+x+10+np.random.normal(0,1,size=100)
    # 增加过多特征见证过拟合效果
    x2 = np.hstack([x1, x1 ** 2,x1 ** 3,x1 ** 4,x1 ** 5,x1 ** 6,x1 ** 7,x1 ** 8,x1 ** 9,x1 ** 10,x1 ** 11])

    model = LinearRegression()
    model.fit(x2,y)
    y_p = model.predict(x2)
    print(root_mean_squared_error(y_p, y))
    print(mean_squared_error(y_p, y))
    print(mean_absolute_error(y_p, y))

    plt.scatter(x, y)
    plt.plot(x[np.argsort(x)], y_p[np.argsort(x)],color='red')
    plt.show()

def dm_模型拟合_过拟合处理_l1():
    np.random.seed(18)
    x = np.random.uniform(-3,3,size=100)
    x1 = x.reshape(-1, 1)
    y = 0.5*x**2+x+10+np.random.normal(0,1,size=100)

    x2 = np.hstack([x1, x1 ** 2,x1 ** 3,x1 ** 4,x1 ** 5,x1 ** 6,x1 ** 7,x1 ** 8,x1 ** 9,x1 ** 10,x1 ** 11])
    # 调用L1正则化,设置惩罚率为0.001
    model = Lasso(alpha=0.001)
    model.fit(x2,y)
    y_p = model.predict(x2)
    print(root_mean_squared_error(y_p, y))
    print(mean_squared_error(y_p, y))
    print(mean_absolute_error(y_p, y))

    plt.scatter(x, y)
    plt.plot(x[np.argsort(x)], y_p[np.argsort(x)],color='red')
    plt.show()

def dm_模型拟合_过拟合处理_l2():
    np.random.seed(18)
    x = np.random.uniform(-3, 3, size=100)
    x1 = x.reshape(-1, 1)
    y = 0.5 * x ** 2 + x + 10 + np.random.normal(0, 1, size=100)

    x2 = np.hstack([x1, x1 ** 2,x1 ** 3,x1 ** 4,x1 ** 5,x1 ** 6,x1 ** 7,x1 ** 8,x1 ** 9,x1 ** 10,x1 ** 11])
    # 调用L2正则化,设置惩罚率为0.001
    model = Ridge(alpha=0.001)
    model.fit(x2, y)
    y_p = model.predict(x2)
    print(root_mean_squared_error(y_p, y))
    print(mean_squared_error(y_p, y))
    print(mean_absolute_error(y_p, y))

    plt.scatter(x, y)
    plt.plot(x[np.argsort(x)], y_p[np.argsort(x)], color='red')
    plt.show()
if __name__ == '__main__':
    dm_模型拟合_过拟合处理_l1()
