import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter  # 计数器
from sklearn.metrics import accuracy_score  # 模型评估,计算模型预测准确率
import warnings
from sklearn.preprocessing import StandardScaler

from knn算法.knn分类实现 import x_train
# 压制警告
warnings.filterwarnings("ignore", module="sklearn")

def show_num(idx):
    # 打印出数字图像
    # 导入数据
    df = pd.read_csv(r'../data/手写数字识别.csv')
    # 防止索引越界
    if idx < 0 or idx > len(df) - 1:
        print("索引越界!")
        return
    # 获取数据
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    # 转为(28,28)形状的矩阵
    x = x[x.index==idx].values.reshape(28,28)
    # 绘制图形
    plt.imshow(x,cmap='gray')
    plt.show()
    print(x.shape)

def model_c():
    # 模型创建
    # 导入数据
    df = pd.read_csv(r'../data/手写数字识别.csv')
    # 数据处理
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    x=x/255
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=18,stratify=y)
    #实例标准化对象
    # trans = StandardScaler()
    # # 标准化训练集和测试集特征数据
    # x_train=trans.fit_transform(x_train)
    # x_test=trans.transform(x_test)
    # 创建模型对象
    model = KNeighborsClassifier(3)
    # 训练模型
    model.fit(x_train,y_train)

    # 模型预测
    print(f'准确率{model.score(x_test, y_test)}')
    print(f'准确率{accuracy_score(y_test,model.predict(x_test))}')
    # 将模型保存为文件
    joblib.dump(model, '../model/手写数字识别.pkl')

def model_u():
    # 调用模型文件
    k_model = joblib.load('../model/手写数字识别.pkl')
    # 图片转为数字数据并降维到一维数组
    img = plt.imread('../data/demo_0.png').reshape(1,-1)
    print(img)
    # 用模型预测结果
    print(k_model.predict(img))

if __name__ == '__main__':
    model_u()