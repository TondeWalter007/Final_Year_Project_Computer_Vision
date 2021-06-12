import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

image_list = ['41A2.jpg', '41A3.jpg', '41A4.jpg', '41A5.jpg', '41B1.jpg', '41B2.jpg', '41B3.jpg', '41B4.jpg',
              '41B5.jpg', '42A1.jpg', '42A2.jpg', '42A3.jpg', '42A4.jpg', '42A5.jpg', '42B1.jpg', '42B2.jpg',
              '42B3.jpg', '42B4.jpg', '42B5.jpg', '43A1.jpg', '43A2.jpg', '43A3.jpg', '43A4.jpg', '43A5.jpg',
              '43B1.jpg', '43B3.jpg', '43B4.jpg', '43B5.jpg', '61A1.jpg', '61A2.jpg', '61A3.jpg', '61A4.jpg',
              '61A5.jpg', '61B1.jpg', '61B2.jpg', '61B3.jpg', '61B4.jpg', '61B5.jpg', '62A1.jpg', '62A2.jpg',
              '62A3.jpg', '62A4.jpg', '62A5.jpg', '62B1.jpg', '62B2.jpg', '62B3.jpg', '62B4.jpg', '62B5.jpg',
              '63A1.jpg', '63A3.jpg', '63A4.jpg', '63A5.jpg', '63B1.jpg', '63B2.jpg', '63B3.jpg', '63B4.jpg',
              '63B5.jpg', '81A1.jpg', '81A3.jpg', '81A4.jpg', '81A5.jpg', '81B1.jpg', '81B2.jpg',
              '81B3.jpg', '81B4.jpg', '81B5.jpg', '82A1.jpg', '82A2.jpg', '82A3.jpg', '82A4.jpg', '82A5.jpg',
              '82B1.jpg', '82B2.jpg', '82B3.jpg', '82B4.jpg', '83A1.jpg', '83A2.jpg', '83A3.jpg', '83A4.jpg',
              '83A5.jpg', '83B1.jpg', '83B2.jpg', '83B3.jpg', '83B4.jpg', '83B5.jpg']

l = [43.36, 44.21, 34.36, 37.9, 54.34, 57.42, 58.76, 63.04, 56.3, 46.4, 52.51, 52.76, 40.77, 38.42,
     53.64, 45.3, 61.17, 59.62, 63.19,
     49.88, 49.25, 44.99, 49.38, 40.33, 39.21, 43.71,
     48.76, 52.09, 54.36, 48.94, 51.25, 50.36, 48.75, 52.61]

a = [9.69, 10.15, 9.21, 4.93, 8.1, 5.98, 5.29, 7.11,
     4.61, 6.31, 3.5, 4.51, 5.15, 3.84, 4.89,
     5.1, 8.54, 6.85, 4.62, 4.0, 6.29, 2.18,
     5.23, 9.52, 10.04, 9.89, 12.3, 6.7, 6.13, 7.73, 6.39,
     10.65, 5.67, 5.38]

b = [16.29, 12.07, 13.4, 9.59, 4.22, 6.4, 4.32, 4.97,
     5.3, 6.71, 4.93, 5.92, 5.1, 6.58, 6.05,
     6.27, 9.1, 9.08, 3.44, 6.39, 7.89, 3.93,
     6.86, 11.96, 12.12, 11.35, 9.7, 6.15, 7.72, 14.61,
     7.27, 10.21, 10.14, 7.47]

gsi = [9.82, 7.49, 9.13, 11.17, 6.45, 16.53, 12.56, 15.46, 18.22, 18.96, 22.87, 9.04, 11.47, 10.41, 12.23, 12.27, 8.41,
       15.31, 9.23, 13.34, 17.61, 15.02, 8.50, 15.8, 16.09, 15.75, 10.21, 12.96, 13.01, 11.86, 16.62, 4.57, 10.15,
       13.01]

color_corr = {
    'l': [43.36, 44.21, 34.36, 37.9, 54.34, 57.42, 58.76, 63.04, 56.3, 46.4, 52.51, 52.76, 40.77, 38.42,
          53.64, 45.3, 61.17, 59.62, 63.19,
          49.88, 49.25, 44.99, 49.38, 40.33, 39.21, 43.71,
          48.76, 52.09, 54.36, 48.94, 51.25, 50.36, 48.75, 52.61],
    'a': [9.69, 10.15, 9.21, 4.93, 8.1, 5.98, 5.29, 7.11,
          4.61, 6.31, 3.5, 4.51, 5.15, 3.84, 4.89,
          5.1, 8.54, 6.85, 4.62, 4.0, 6.29, 2.18,
          5.23, 9.52, 10.04, 9.89, 12.3, 6.7, 6.13, 7.73, 6.39,
          10.65, 5.67, 5.38],
    'b': [16.29, 12.07, 13.4, 9.59, 4.22, 6.4, 4.32, 4.97,
          5.3, 6.71, 4.93, 5.92, 5.1, 6.58, 6.05,
          6.27, 9.1, 9.08, 3.44, 6.39, 7.89, 3.93,
          6.86, 11.96, 12.12, 11.35, 9.7, 6.15, 7.72, 14.61,
          7.27, 10.21, 10.14, 7.47],
    'gsi': [9.82, 7.49, 9.13, 11.17, 6.45, 16.53, 12.56, 15.46, 18.22, 18.96, 22.87, 9.04, 11.47, 10.41, 12.23, 12.27,
            8.41, 15.31, 9.23, 13.34, 17.61, 15.02, 8.50, 15.8, 16.09, 15.75, 10.21, 12.96, 13.01, 11.86, 16.62, 4.57,
            10.15, 13.01]
}

print('length of l array:', len(l))
print('length of a array:', len(a))
print('length of b array:', len(b))
print('length of gsi array:', len(gsi))
print('')
print('average of l array:', np.average(l))
print('average of a array:', np.average(a))
print('average of b array:', np.average(b))
print('average of gsi array:', np.average(gsi))

print("")

df = pd.DataFrame(color_corr, columns=['l', 'a', 'b', 'gsi'])
print(df)

plt.scatter(df['b'], df['gsi'], color='red')
plt.title('B Vs GSI', fontsize=14)
plt.xlabel('B', fontsize=14)
plt.ylabel('GSI', fontsize=14)
plt.grid(True)
plt.show()

X = df[['l', 'a', 'b']]
Y = df['gsi']

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
