import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
data=pd.read_csv("dự đoán giá nhà/diabetes.csv")
# print(type(data))
x=data.drop("Outcome",axis=1)
y=data["Outcome"]
# print(x)
# print(y)
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

cls = SVC()
cls.fit(x_train, y_train)

y_predict = cls.predict(x_test)
# for i,j in zip(y_test, y_predict):
#     print(f"Actual: {i}, Predicted: {j}")
print(classification_report(y_test, y_predict,))
