import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

l_spine = [43.36, 44.21, 34.36, 37.9, 54.34, 57.42, 58.76, 63.04, 56.3, 46.4, 52.51, 52.76, 40.77, 49.25, 53.64, 45.3,
           61.17, 59.62, 63.19, 49.88, 49.25, 44.99, 49.38, 40.33, 39.21, 43.71, 48.76, 52.09, 54.36, 48.94, 51.25,
           50.36, 48.75, 52.61]

a_spine = [9.69, 10.15, 9.21, 4.93, 8.1, 5.98, 5.29, 7.11, 4.61, 6.31, 3.5, 4.51, 5.15, 6.29, 4.89, 5.1, 8.54,
           6.85, 4.62, 4.0, 6.29, 2.18, 5.23, 9.52, 10.04, 9.89, 12.30, 6.7, 6.13, 7.73, 6.39, 10.65, 5.67, 5.38]

b_spine = [16.29, 12.07, 13.4, 9.59, 4.22, 6.4, 4.32, 4.97, 5.3, 6.71, 4.93, 5.92, 5.1, 7.89, 6.05, 6.27, 9.1,
           9.08, 3.44, 6.39, 7.89, 3.93, 6.86, 11.96, 12.12, 11.35, 9.70, 6.15, 7.27, 14.64, 7.27, 10.21, 10.14, 7.47]

print('length of l_spine array:', len(l_spine))
print('length of a_spine array:', len(a_spine))
print('length of b_spine array:', len(b_spine))

print('')
print('average of l_spine array:', np.average(l_spine))
print('average of a_spine array:', np.average(a_spine))
print('average of b_spine array:', np.average(b_spine))

print("")

l_gonad = [65.67, 51.7, 53.57, 60.53, 50.9, 54.83, 58.67, 53.97, 50.43, 49.9, 50.6, 52.03, 54.73, 59.1, 50.83, 48.5,
           59.03, 55.93, 50.37, 53.2, 56.77, 58.23, 55.77, 52.63, 67.2, 55.6, 62.37, 57.03, 58.00, 58.37, 63.1, 55.3,
           61.4, 55.2]

a_gonad = [13.37, 8.73, 14.1, 6.67, 11.43, 16.9, 11.9, 15.63, 16.63, 16.9, 13.7, 15.07, 10.3, 12.33, 6.4, 12.3, 10.9,
           13.83, 9.57, 8.87, 13.02, 12.87, 14.2, 18.07, 7.53, 18.17, 10.77, 13.17, 7.53, 14.00, 6.8, 10.9, 12.57, 15.3]

b_gonad = [37.5, 26.33, 37.63, 32.3, 32.97, 36.6, 35.2, 37.23, 38.87, 40.23, 35.2, 37.03, 33.13, 33.9, 31.77, 32.33,
           32.87, 35.13, 34.43, 34.8, -1.83, 41.77, 39.83, 37.6, 31.47, 42.93, 36.37, 40.9, 33.3, 38.37, 34.5, 35.97,
           35.57, 37.6]

print('length of l_gonad array:', len(l_gonad))
print('length of a_gonad array:', len(a_gonad))
print('length of b_gonad array:', len(b_gonad))

print('')
print('average of l_gonad array:', np.average(l_gonad))
print('average of a_gonad array:', np.average(a_gonad))
print('average of b_gonad array:', np.average(b_gonad))

print("")

l_diff = [22.31, 7.49, 19.21, 22.63, -3.44, -2.59, 0.09, -9.07, -5.87, 3.5, -1.91, -0.73, 13.96, 9.85, -2.81, 3.20,
          -2.14, -3.69, -12.82, 3.32, 7.52, 13.24, 6.39, 12.30, 27.99, 11.89, 13.61, 4.94, 3.64, 9.43, 11.85, 4.94,
          12.65, 2.59]

a_diff = [3.78, -1.42, 4.89, 1.74, 3.33, 10.92, 6.61, 8.52, 12.02, 10.59, 10.2, 10.56, 5.15, 6.04, 1.51, 7.20, 2.36,
          6.98, 4.95, 4.87, 6.74, 10.69, 8.97, 8.55, -2.51, 8.28, -1.53, 6.47, 1.4, 6.27, 0.41, 0.25, 6.9, 9.92]

b_diff = [21.21, 14.26, 24.23, 22.71, 28.75, 30.20, 30.88, 32.26, 33.57, 33.52, 30.27, 31.11, 28.03, 26.01, 25.72,
          26.06, 23.77, 26.05, 30.99, 28.41, -9.72, 37.84, 32.97, 25.64, 19.35, 31.58, 26.67, 34.75, 25.58, 23.76,
          27.23, 25.76, 25.53, 30.13]

print('length of l_diff array:', len(l_diff))
print('length of a_diff array:', len(a_diff))
print('length of b_diff array:', len(b_diff))

print('')
print('average of l_diff array:', np.average(l_diff))
print('average of a_diff array:', np.average(a_diff))
print('average of b_diff array:', np.average(b_diff))

print("")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a_diff, b_diff, l_diff)
ax.set_xlabel("A (red-blue)")
ax.set_ylabel("B (green-yellow)")
ax.set_zlabel("L (lightness)")
plt.title('LAB 3D Scatter Plot')
plt.xlim([-50, 50])
plt.ylim([-50, 50])
ax.set_zlim(-50, 50)
plt.show()

color_corr = {
    'l_spine': [43.36, 44.21, 34.36, 37.9, 54.34, 57.42, 58.76, 63.04, 56.3, 46.4, 52.51, 52.76, 40.77, 49.25, 53.64,
                45.3,
                61.17, 59.62, 63.19, 49.88, 49.25, 44.99, 49.38, 40.33, 39.21, 43.71, 48.76, 52.09, 54.36, 48.94, 51.25,
                50.36, 48.75, 52.61],
    'a_spine': [9.69, 10.15, 9.21, 4.93, 8.1, 5.98, 5.29, 7.11, 4.61, 6.31, 3.5, 4.51, 5.15, 6.29, 4.89, 5.1, 8.54,
                6.85, 4.62, 4.0, 6.29, 2.18, 5.23, 9.52, 10.04, 9.89, 12.30, 6.7, 6.13, 7.73, 6.39, 10.65, 5.67, 5.38],
    'b_spine': [16.29, 12.07, 13.4, 9.59, 4.22, 6.4, 4.32, 4.97, 5.3, 6.71, 4.93, 5.92, 5.1, 7.89, 6.05, 6.27, 9.1,
                9.08, 3.44, 6.39, 7.89, 3.93, 6.86, 11.96, 12.12, 11.35, 9.70, 6.15, 7.27, 14.64, 7.27, 10.21, 10.14,
                7.47],
    'l_gonad': [65.67, 51.7, 53.57, 60.53, 50.9, 54.83, 58.67, 53.97, 50.43, 49.9, 50.6, 52.03, 54.73, 59.1, 50.83,
                48.5, 59.03, 55.93, 50.37, 53.2, 56.77, 58.23, 55.77, 52.63, 67.2, 55.6, 62.37, 57.03, 58.00, 58.37,
                63.1, 55.3, 61.4, 55.2],
    'a_gonad': [13.37, 8.73, 14.1, 6.67, 11.43, 16.9, 11.9, 15.63, 16.63, 16.9, 13.7, 15.07, 10.3, 12.33, 6.4, 12.3,
                10.9, 13.83, 9.57, 8.87, 13.02, 12.87, 14.2, 18.07, 7.53, 18.17, 10.77, 13.17, 7.53, 14.00, 6.8, 10.9,
                12.57, 15.3],
    'b_gonad': [37.5, 26.33, 37.63, 32.3, 32.97, 36.6, 35.2, 37.23, 38.87, 40.23, 35.2, 37.03, 33.13, 33.9, 31.77,
                32.33, 32.87, 35.13, 34.43, 34.8, -1.83, 41.77, 39.83, 37.6, 31.47, 42.93, 36.37, 40.9, 33.3, 38.37,
                34.5, 35.97, 35.57, 37.6]
}

df = pd.DataFrame(color_corr, columns=['l_spine', 'a_spine', 'b_spine', 'l_gonad', 'a_gonad', 'b_gonad'])
print(df)

X_l = df[['l_gonad']]
Y_l = df['l_spine']

X_a = df[['a_gonad']]
Y_a = df['a_spine']

X_b = df[['b_gonad']]
Y_b = df['b_spine']

# red dashes, blue squares and green triangles
plt.plot(l_gonad, l_spine, 'r--', a_gonad, a_spine, 'bs', b_gonad, b_spine, 'g^')
plt.show()

# Lightness correlation
reg_l = linear_model.LinearRegression()
reg_l.fit(X_l, Y_l)

print("")
print("Lightness:")
print('Intercept: \n', reg_l.intercept_)
print('Coefficients: \n', reg_l.coef_)
print("")

# with statsmodels
X_l = sm.add_constant(X_l)  # adding a constant

model = sm.OLS(Y_l, X_l).fit()
predictions = model.predict(X_l)

print_model = model.summary()
print(print_model)
