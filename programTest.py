import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics

manual = [51.43, 6.05, 9.11]
auto = [52.61, 5.38, 7.47]

L_num = manual[0] - auto[0]
A_num = manual[1] - auto[1]
B_num = manual[2] - auto[2]

print("L:", L_num)
print("A:", A_num)
print("B:", B_num)

L_diff = [2.98, 1.94, 2.04, 2.19, 1.71, 1.59, 1.12, 1.65, 1.90, 1.48, 2.10, 1.49, 1.40, 1.68, 1.45, 1.70, 2.02, 1.30, -0.76,
     1.54, 1.94, 0.92, 1.85, -1.07, 1.36, 1.40, 1.36, 1.41, 1.95, 0.96, 1.33, 1.93, 1.84, 1.83, -1.18]
A_diff = [1.06, 1.34, 1.33, 1.91, 0.57, 0.89, 1.06, 1.17, 1.00, 0.92, 1.07, 1.08, 1.34, 1.60, 1.77, 1.24, 1.39, 1.22, 0.70,
     1.57, 1.58, 1.89, 1.42, 1.48, 1.18, 1.43, 1.10, 1.34, 0.75, 1.65, 1.48, 1.09, 1.02, 0.83, 0.67]
B_diff = [1.17, 1.20, -1.35, 1.22, 0.90, 1.36, 1.91, 0.56, 1.08, 0.86, 0.41, 1.00, 1.02, 0.74, 1.04, 1.39, 0.94, 1.06, 1.55,
     1.26, 1.10, 1.27, 1.10, 1.28, 1.91, 1.27, 1.93, 1.57, 1.06, 1.15, 1.22, 0.88, 1.12, 1.03, 1.64]

Delta_E = [3.28, 2.65, 2.74, 3.08, 2.05, 2.20, 2.33, 2.10, 2.34, 1.91, 2.37, 2.03, 2.12, 2.39, 2.43, 2.46, 2.54, 2.00,
           2.41, 1.75, 2.51, 2.35, 2.17, 2.46, 2.54, 2.35, 2.51, 2.40, 2.25, 2.13, 2.29, 2.31, 2.32, 2.21, 2.08]

print("Length of L:", len(L_diff))
print("Length of A:", len(A_diff))
print("Length of B:", len(B_diff))
print("Length of Delta_E:", len(Delta_E))
print("Percentage of Negatives:", round((3 / 35 * 100), 2))
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

print("Count =", count)
std_w = len(Delta_E) - count
print("Percentage of Values within Standard Deviation:", (round((std_w/len(Delta_E)*100), 2)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A_diff, B_diff, L_diff)
ax.set_xlabel("A (red-blue)")
ax.set_ylabel("B (green-yellow)")
ax.set_zlabel("L (lightness)")
plt.title('LAB 3D Scatter Plot')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
ax.set_zlim(-3, 3)
plt.show()
