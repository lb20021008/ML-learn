import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV  # 分割训练集和测试集  寻找最优的超参
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估,计算模型预测准确率
# 获取数据
iris_data = load_iris()
# 分割训练集和测试集比例8:2,随机种子为9
x_train, x_test, y_train, y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=9)
# 标准化训练集测试集特征
transfer=StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)
# 实例化模型
model=KNeighborsClassifier()
# 超参范围准备
pram_dict={
    'n_neighbors':[i for i in range(2,11)]
}
# 传入模型,超参范围,折数返回模型
model = GridSearchCV(model, param_grid=pram_dict,cv=5)
# 训练模型
model.fit(x_train,y_train)
# 打印最优评分,超参组合
print(f"最优评分:{model.best_score_}")
print(f"最优超参组合:{model.best_params_}")
print(f"最优估计器对象:{model.best_estimator_}")

# 使用最优超参构建模型
model2=KNeighborsClassifier(model.best_params_['n_neighbors'])
model2.fit(x_train,y_train)
print(model2.predict(x_test))
print(y_test)
print(model2.score(x_test,y_test))

