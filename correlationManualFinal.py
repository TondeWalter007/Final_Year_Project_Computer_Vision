# DONT USE

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

l_man_fin = [46.34, 46.15, 36.40, 40.09, 56.06, 59.01, 59.88, 64.69, 58.20, 47.88, 54.61, 54.25, 42.17, 40.10, 55.09,
             47.00, 63.19, 54.16, 58.86, 64.73, 51.82, 50.17, 46.84, 48.31, 41.69, 40.61, 45.07, 50.17, 54.04, 55.32,
             50.27, 53.18, 52.20, 50.58, 51.43]
a_man_fin = [10.75, 11.49, 10.54, 6.84, 8.67, 6.87, 6.35, 8.28, 5.61, 7.23, 4.57, 5.59, 6.49, 5.44, 6.66, 6.34, 9.93,
             7.41, 7.55, 6.19, 5.58, 8.18, 3.60, 6.71, 10.70, 11.47, 10.99, 13.64, 7.45, 7.78, 9.21, 7.48, 11.67, 6.50,
             6.05]
b_man_fin = [17.46, 13.27, 12.05, 10.81, 5.12, 7.76, 6.23, 5.53, 6.38, 7.57, 5.34, 6.92, 6.12, 7.32, 7.09, 7.66, 10.04,
             8.84, 10.63, 4.70, 7.49, 9.16, 5.03, 8.14, 13.87, 13.39, 13.28, 11.27, 7.21, 8.87, 15.83, 8.15, 11.33,
             11.17, 9.11]

gsi = [9.82, 7.49, 9.13, 11.17, 6.44, 16.53, 12.56, 15.46, 18.22, 18.96, 22.87, 9.04, 11.47, 10.41, 12.27, 12.23,
       8.41, 10.30, 9.23, 15.31, 13.34, 17.61, 8.50, 15.02, 15.80, 16.09, 15.75, 10.21, 12.96, 13.01, 11.86, 16.62,
       4.57, 13.01, 7.36]

color_corr = {
    'l': [46.34, 46.15, 36.40, 40.09, 56.06, 59.01, 59.88, 64.69, 58.20, 47.88, 54.61, 54.25, 42.17, 40.10, 55.09,
          47.00, 63.19, 54.16, 58.86, 64.73, 51.82, 50.17, 46.84, 48.31, 41.69, 40.61, 45.07, 50.17, 54.04, 55.32,
          50.27, 53.18, 52.20, 50.58, 51.43],
    'a': [10.75, 11.49, 10.54, 6.84, 8.67, 6.87, 6.35, 8.28, 5.61, 7.23, 4.57, 5.59, 6.49, 5.44, 6.66, 6.34, 9.93,
          7.41, 7.55, 6.19, 5.58, 8.18, 3.60, 6.71, 10.70, 11.47, 10.99, 13.64, 7.45, 7.78, 9.21, 7.48, 11.67, 6.50,
          6.05],
    'b': [17.46, 13.27, 12.05, 10.81, 5.12, 7.76, 6.23, 5.53, 6.38, 7.57, 5.34, 6.92, 6.12, 7.32, 7.09, 7.66, 10.04,
          8.84, 10.63, 4.70, 7.49, 9.16, 5.03, 8.14, 13.87, 13.39, 13.28, 11.27, 7.21, 8.87, 15.83, 8.15, 11.33,
          11.17, 9.11]
    ,
    'gsi': [9.82, 7.49, 9.13, 11.17, 6.44, 16.53, 12.56, 15.46, 18.22, 18.96, 22.87, 9.04, 11.47, 10.41, 12.27, 12.23,
            8.41, 10.30, 9.23, 15.31, 13.34, 17.61, 8.50, 15.02, 15.80, 16.09, 15.75, 10.21, 12.96, 13.01, 11.86, 16.62,
            4.57, 13.01, 7.36]
}

print('length of l array:', len(l_man_fin))
print('length of a array:', len(a_man_fin))
print('length of b array:', len(b_man_fin))
print('length of gsi array:', len(gsi))
print('')
print('average of l array:', np.average(l_man_fin))
print('average of a array:', np.average(a_man_fin))
print('average of b array:', np.average(b_man_fin))
print('average of gsi array:', np.average(gsi))

print("")

df = pd.DataFrame(color_corr, columns=['l', 'a', 'b', 'gsi'])
print(df)

#plt.scatter(df['b'], df['gsi'], color='red')
#plt.title('B Vs GSI', fontsize=14)
#plt.xlabel('B', fontsize=14)
#plt.ylabel('GSI', fontsize=14)
#plt.grid(True)
#plt.show()

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
