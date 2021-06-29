import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import math

df = pd.read_csv('C:/Users/Admin/Desktop/EEE4022F/Gonad_Color_and_GSI.csv')

print(df)

X = df[['L*', 'a*', 'b*']]
Y = df['GSI']

# linear regression model
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)