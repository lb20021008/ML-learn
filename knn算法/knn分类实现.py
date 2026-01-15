# 导包knn分类算法
from sklearn.neighbors import KNeighborsClassifier
# 构建模型,k值为3
model = KNeighborsClassifier(n_neighbors=3)
# 构建训练集
x_train = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
y_train = [1,1,1,1,1,2,2,2,2,2,3,3,3,4,4]
# 训练模型
model.fit(x_train, y_train)
# 构建测试集样本
x_test = [[10]]
# 得到测试集标签
y_test = model.predict(x_test)
# 打印输出结果哦
print(y_test)