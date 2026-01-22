from sklearn.linear_model import LogisticRegression,SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,roc_auc_score

# 案例分析,数据中年费用户和月费用户互补只留一列即可,互联网使用情况同样如此
df = pd.read_csv('../data/churn.csv',usecols=['Churn','gender','Dependents_att','landline','internet_att','StreamingTV','StreamingMovies','Contract_Month','PaymentCreditcard','PaymentElectronic','MonthlyCharges','TotalCharges'])

# df.info()

df.replace('No',0,inplace=True)
df.replace('Yes',1,inplace=True)
df.replace('Female',0,inplace=True)
df.replace('Male',1,inplace=True)

# 数据处理
x = df.iloc[:,1:]
y = df.iloc[:,0]

# 训练集数据集划分
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=18,stratify=y)

# 处理特征量纲问题
sc = StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
random_state_d = {'random_state':[i for i in range(200)]}
# 建模逻辑回归
model = LogisticRegression()
# 交叉验证和网格搜索找出最好超参和模型
model = GridSearchCV(model,param_grid=random_state_d,cv=5)
model.fit(x_train,y_train)
# 打印评分
print(f"最优评分:{model.best_score_}")
print(f"最优超参组合:{model.best_params_}")
print(f"最优估计器对象:{model.best_estimator_}")
# 用找出来的最好模型进行预测
y_pred = model.best_estimator_.predict(x_test)
# 预测结果和评估分数
print(y_pred)
print(recall_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
print(model.best_estimator_.intercept_)
print(model.best_estimator_.coef_)
print(model.best_estimator_.score(x_test,y_test))
# 打印各特征对应权重值,分析特征与结果相关程度
w = pd.DataFrame(model.best_estimator_.coef_)
w = w.T
# 按权重值排序,权重绝对值越大代表对预测结果影响越大
print(w.sort_values(by=[0],ascending=False))

# 查看特征名分析
print(x.columns)


# 未使用互联网,月度用户,互联网用户因素