from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score
import numpy as np
import pandas as pd
#准备训练集测试集
y_train = ['恶性','恶性','恶性','恶性','恶性','恶性','良性','良性','良性','良性']

y_pred_A = ['恶性','恶性','恶性','良性','良性','良性','良性','良性','良性','良性']

y_pred_B = ['恶性','恶性','恶性','恶性','恶性','恶性','恶性','恶性','恶性','良性']

labels_c = ['恶行','良性']
# 给混淆矩阵设置行列名
df_labels = ['恶行(正例)','良性(反例)']
# 混淆矩阵a
c_a = confusion_matrix(y_train,y_pred_A)
# 混淆矩阵b
c_b = confusion_matrix(y_train,y_pred_B)
# 转为dataframe格式
df_ca = pd.DataFrame(c_a,index=labels_c,columns=labels_c)
print(df_ca)

df_cb = pd.DataFrame(c_b,index=labels_c,columns=labels_c)
print(df_cb)
# 查看指标
print(f'A精确率{precision_score(y_train,y_pred_A,pos_label='恶性')}')
print(f'B精确率{precision_score(y_train,y_pred_B,pos_label='恶性')}')

print(f'A精确率{recall_score(y_train,y_pred_A,pos_label='恶性')}')
print(f'B精确率{recall_score(y_train,y_pred_B,pos_label='恶性')}')

print(f'A精确率{f1_score(y_train,y_pred_A,pos_label='恶性')}')
print(f'B精确率{f1_score(y_train,y_pred_B,pos_label='恶性')}')