from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor # 梯度下降
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# from sklearn.datasets import load_boston

# 1.获取数据
df = pd.read_csv('../data/boston.csv')
# 线上获取数据集
# datas = load_boston()
# print(datas)
# 特征提取
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']
tar = ['MEDV']

datas = df[cols]
targets = df[tar]

# 训练集,测试集划分
x_train,x_test,y_train,y_test = train_test_split(datas,targets,test_size=0.2,random_state=666)

# 特征与处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 模型训练,使用正规方程方法
model = SGDRegressor(fit_intercept=True,random_state=18,eta0=0.01)
model.fit(x_train,y_train)

# 预测
y_pred = model.predict(x_test)

# 评估
print(f'预测值{y_pred}')
print(f'实际值{y_test}')
print(f"权重{model.coef_}")
print(f"偏置{model.intercept_}")

print(f'平均绝对误差{mean_absolute_error(y_test, y_pred)}')
print(f'均方误差{mean_squared_error(y_test, y_pred)}')
print(f'均方根误差{root_mean_squared_error(y_test, y_pred)}')
