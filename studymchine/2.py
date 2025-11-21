import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv("student_data_test.csv")
# print(type(data))
x=data.drop("Outcome",axis=1)
y=data["Outcome"]
# print(x)
# print(y)
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
