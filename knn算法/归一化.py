from  sklearn.preprocessing import MinMaxScaler
# 数据准备
data = [
    [90, 2, 10, 40],
     [60, 4, 15, 45],
     [75, 3, 13, 46]
]
# 实例化归一化模型,mi=0,mx=1
tans = MinMaxScaler(feature_range=(0, 1))
# 对数据进行归一化
new_data = tans.fit_transform(data)
print(new_data)