# 导入knn回归模型包
from sklearn.neighbors import KNeighborsRegressor
# 设置训练集样本
x_train = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
]
# 给定训练集标签
y_train = [1,1.1,1.4,1.7,2]
# 给定测试集样本
x_test = [
    [3,1,7]
]
# 创建模型对象,设置k为3
model = KNeighborsRegressor(n_neighbors=3)
# 训练模型
model.fit(x_train, y_train)
# 打印输出测试集结果\标签
print(model.predict(x_test))