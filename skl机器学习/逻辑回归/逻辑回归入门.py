from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
df = pd.read_csv('../data/breast-cancer-wisconsin.csv')
df.replace('?',np.NAN,inplace=True)
df.dropna(inplace=True)
x = df.iloc[:,1:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66666)
stc =StandardScaler()
x_train=stc.fit_transform(x_train)
x_test=stc.transform(x_test)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
asce = accuracy_score(y_test,y_pred)
print(f'准确率{asce},预测结果{y_pred}')