import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import seaborn as sns
def load_iris_data():
    # 数据分析
    data_l = load_iris()
    print(data_l)
    # 查看特征数据
    # print(data_l.data)
    # 查看标签分类
    # print(data_l.target)
    # 特征名
    # print(data_l.feature_names)
    # 标签名
    # print(data_l.target_names)
def dm02_show_iris():
    # 鸢尾花散点图
    data_l = load_iris()
    # 取data(特征数据),和特征名构造df对象
    df_l = pd.DataFrame(data_l.data, columns=data_l.feature_names)
    # 添加标签列
    df_l['target'] = data_l.target
    # 绘制散点图传入df对象,x轴取花瓣长度petal length (cm),y轴取花瓣宽度petal width (cm) hue='target'表示通过标签区分颜色,fit_reg=False表示不画出回归线
    sns.lmplot(data=df_l,x='petal length (cm)',y='petal width (cm)',hue='target',fit_reg=False)
    plt.show()
def dm03_train_tet_iris():
    # 获取训练集测试集
    data_l = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(data_l.data,data_l.target,test_size=0.2,random_state=10)
    print(x_train)
    print('*'*40)
    print(y_train)
    print('*'*40)
    print(x_test)
    print('*'*40)
    print(y_test)
def dm04_模型实现():
    # 实现模型
    # 读取数据
    data_l = load_iris()
    # 获取训练集,测试集
    x_train, x_test, y_train, y_test = train_test_split(data_l.data, data_l.target, test_size=0.2, random_state=10)
    # 实例话标准化对象
    trans = StandardScaler()
    # 对训练集和测试集数据进行标准化
    x_train=trans.fit_transform(x_train)
    x_test=trans.transform(x_test)
    # 实例化模型对象
    model = KNeighborsClassifier(n_neighbors=5)
    # 训练模型
    model.fit(x_train,y_train)
    # 预测结果
    y_pred = model.predict(x_test)
    # 对比结果
    print(f'预测结果{y_pred}')
    print(f'真实结果{y_test}')
    # 打印准确率
    accuracy = accuracy_score(y_test,y_pred)
    # model.score(x_test,y_test) 第二种方式
    print('准确率:',accuracy)
if __name__ == '__main__':
    dm04_模型实现()