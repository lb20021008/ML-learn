import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report # 模型综合评测,包含准确率,精准率,召回率,f1值
from sklearn.tree import DecisionTreeClassifier,plot_tree # 决策树包,用来画决策树的包
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 导入数据
df = pd.read_csv('../data/train.csv')
# 选用指定特征
x = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df['Survived']
# 缺失值处理
x['Age'] = x['Age'].fillna(x['Age'].mean())
# 对二分类进行热编码
x = pd.get_dummies(x)
# 删除多余的性别列
x.drop('Sex_female',axis=1,inplace=True)
# 划分训练集测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
# 特征预处理
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# 模型实例化
model = DecisionTreeClassifier()
# 超参范围
max_depth_p ={'max_depth': [i for i in range(5,40)],
              'min_samples_leaf':[i for i in range(1,10)],}
# 交叉验证网格搜索
model_b = GridSearchCV(model,param_grid=max_depth_p,cv=5)
# 获取最优评分,最优超参,最优模型对象
model_b.fit(x_train,y_train)
print(model_b.best_params_)
print(model_b.best_score_)
print(model_b.best_estimator_)
# 采用最有超参实例化模型并训练
model = DecisionTreeClassifier(max_depth=8,min_samples_leaf=9)
model.fit(x_train,y_train)
# 预测
y_p = model.predict(x_test)
# 打印评分和预测值
print(y_p)
print(classification_report(y_test,y_p))
print(model.score(x_test,y_test))
# 画图
plt.figure(figsize=(100,50))
plot_tree(model,filled=True,max_depth=20)
plt.savefig('../data/tree.png')
plt.show()