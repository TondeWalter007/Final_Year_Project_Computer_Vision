import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics

gonad = [55.2, 15.3, 37.6]
spine = [52.61, 5.38, 7.47]

L_num = gonad[0] - spine[0]
A_num = gonad[1] - spine[1]
B_num = gonad[2] - spine[2]

#print("L:", L_num)
#print("A:", A_num)
#print("B:", B_num)

L_diff = [22.31, 7.49, 19.21, 22.63, -3.44, -2.59, 0.09, -9.07, -5.87, 3.5, -1.91, -0.73, 13.96, 9.85, -2.81, 3.20,
          -2.14, -3.69, -12.82, 3.32, 7.52, 13.24, 6.39, 12.30, 27.99, 11.89, 13.61, 4.94, 3.64, 9.43, 11.85, 4.94,
          12.65, 2.59]

A_diff = [3.78, -1.42, 4.89, 1.74, 3.33, 10.92, 6.61, 8.52, 12.02, 10.59, 10.2, 10.56, 5.15, 6.04, 1.51, 7.20, 2.36,
          6.98, 4.95, 4.87, 6.74, 10.69, 8.97, 8.55, -2.51, 8.28, -1.53, 6.47, 1.4, 6.27, 0.41, 0.25, 6.9, 9.92]

B_diff = [21.21, 14.26, 24.23, 22.71, 28.75, 30.20, 30.88, 32.26, 33.57, 33.52, 30.27, 31.11, 28.03, 26.01, 25.72,
          26.06, 23.77, 26.05, 30.99, 28.41, -9.72, 37.84, 32.97, 25.64, 19.35, 31.58, 26.67, 34.75, 25.58, 23.76,
          27.23, 25.76, 25.53, 30.13]

Delta_E = [30.4, 15.9, 30.5, 31.5, 28.4, 31.0, 30.5, 33.4, 34.9, 34.0, 30.8, 31.6, 30.7, 27.5, 26.2, 25.3, 23.4, 33.1,
           26.3, 28.1, 14.3, 33.5, 40.0, 28.7, 33.9, 33.5, 29.5, 34.5, 25.2, 25.1, 29.0, 25.7, 28.3, 30.6]

print("Length of L:", len(L_diff))
print("Length of A:", len(A_diff))
print("Length of B:", len(B_diff))
print("Length of Delta_E:", len(Delta_E))
#print("Percentage of Negatives:", round((3 / 35 * 100), 2))
print("")

mean_e = statistics.mean(Delta_E)
stddev_e = statistics.stdev(Delta_E)

print("Mean of Delta_E:", round(mean_e, 2))
print("Standard Deviation of Delta_E:", round(stddev_e, 2))

lower_e = mean_e - stddev_e
upper_e = mean_e + stddev_e
count = 0
for i in range(len(Delta_E)):
    if Delta_E[i] < lower_e or Delta_E[i] > upper_e:
        count += 1

print("Number of Points outside Standard Deviation =", count)
std_w = len(Delta_E) - count
print("Percentage of Values within Standard Deviation:", (round((std_w / len(Delta_E) * 100), 2)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A_diff, B_diff, L_diff)
ax.set_xlabel("A (red-blue)")
ax.set_ylabel("B (green-yellow)")
ax.set_zlabel("L (lightness)")
plt.title('LAB 3D Scatter Plot')
plt.xlim([-40, 40])
plt.ylim([-40, 40])
ax.set_zlim(-40, 40)
plt.show()
