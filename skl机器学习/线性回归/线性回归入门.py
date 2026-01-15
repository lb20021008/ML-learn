from sklearn.linear_model import LinearRegression

x = [[160], [166], [172], [174], [180]]
y = [56.3, 60.6, 65.1, 68.5, 75]
model = LinearRegression()
model.fit(x,y)
prediction = model.predict([[176]])
print(prediction)