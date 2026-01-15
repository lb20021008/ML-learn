from sklearn.preprocessing import StandardScaler
# 数据准备
data = [
    [90, 2, 10, 40],
     [60, 4, 15, 45],
     [75, 3, 13, 46]
]
# 实例化标准化模型
tran = StandardScaler()
# 对数据进行标准化
new_data = tran.fit_transform(data)
print(new_data)