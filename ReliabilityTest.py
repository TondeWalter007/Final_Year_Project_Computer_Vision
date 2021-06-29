import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics

L_diff = [2.98, 1.94, 2.04, 2.19, 1.71, 1.59, 1.12, 1.65, 1.90, 1.48, 2.10, 1.49, 1.40, 1.68, 1.45, 1.70, 2.02, 1.30,
          -0.76,
          1.54, 1.94, 0.92, 1.85, -1.07, 1.36, 1.40, 1.36, 1.41, 1.95, 0.96, 1.33, 1.93, 1.84, 1.83, -1.18]
A_diff = [1.06, 1.34, 1.33, 1.91, 0.57, 0.89, 1.06, 1.17, 1.00, 0.92, 1.07, 1.08, 1.34, 1.60, 1.77, 1.24, 1.39, 1.22,
          0.70,
          1.57, 1.58, 1.89, 1.42, 1.48, 1.18, 1.43, 1.10, 1.34, 0.75, 1.65, 1.48, 1.09, 1.02, 0.83, 0.67]
B_diff = [1.17, 1.20, -1.35, 1.22, 0.90, 1.36, 1.91, 0.56, 1.08, 0.86, 0.41, 1.00, 1.02, 0.74, 1.04, 1.39, 0.94, 1.06,
          1.55,
          1.26, 1.10, 1.27, 1.10, 1.28, 1.91, 1.27, 1.93, 1.57, 1.06, 1.15, 1.22, 0.88, 1.12, 1.03, 1.64]

Delta_E = [3.28, 2.65, 2.74, 3.08, 2.05, 2.20, 2.33, 2.10, 2.34, 1.91, 2.37, 2.03, 2.12, 2.39, 2.43, 2.46, 2.54, 2.00,
           2.41, 1.75, 2.51, 2.35, 2.17, 2.46, 2.54, 2.35, 2.51, 2.40, 2.25, 2.13, 2.29, 2.31, 2.32, 2.21, 2.08]


def LAB_diff(L_diff, A_diff, B_diff):
    print("Length of L:", len(L_diff))
    print("Length of A:", len(A_diff))
    print("Length of B:", len(B_diff))
    print("")

    print("L CHANNEL:")
    mean_l = statistics.mean(L_diff)
    stddev_l = statistics.stdev(L_diff)
    print("Lowest L_difference Value:", min(Delta_E))
    print("Highest L_difference Value:", max(Delta_E))
    print("Mean of L_difference Values:", round(mean_l, 2))
    print("Standard Deviation of L_difference Values:", round(stddev_l, 2))

    lower_l = mean_l - stddev_l
    upper_l = mean_l + stddev_l
    count = 0
    for i in range(len(L_diff)):
        if L_diff[i] < lower_l or L_diff[i] > upper_l:
            count += 1

    # print("Number of Delta E Values outside Standard Deviation =", count)
    std_l = len(L_diff) - count
    print("Percentage of L_difference Values within Standard Deviation:", (round((std_l / len(L_diff) * 100), 2)))
    print("")

    print("A CHANNEL:")
    mean_a = statistics.mean(A_diff)
    stddev_a = statistics.stdev(A_diff)
    print("Lowest A_difference Value:", min(A_diff))
    print("Highest A_difference Value:", max(A_diff))
    print("Mean of A_difference Values:", round(mean_a, 2))
    print("Standard Deviation of A_difference Values:", round(stddev_a, 2))

    lower_a = mean_a - stddev_a
    upper_a = mean_a + stddev_a
    count = 0
    for i in range(len(A_diff)):
        if A_diff[i] < lower_a or A_diff[i] > upper_a:
            count += 1

    # print("Number of Delta E Values outside Standard Deviation =", count)
    std_a = len(A_diff) - count
    print("Percentage of A_difference Values within Standard Deviation:", (round((std_a / len(A_diff) * 100), 2)))
    print("")

    print("B CHANNEL:")
    mean_b = statistics.mean(B_diff)
    stddev_b = statistics.stdev(B_diff)
    print("Lowest B_difference Value:", min(B_diff))
    print("Highest B_difference Value:", max(B_diff))
    print("Mean of B_difference Values:", round(mean_b, 2))
    print("Standard Deviation of B_difference Values:", round(stddev_b, 2))

    lower_b = mean_b - stddev_b
    upper_b = mean_b + stddev_b
    count = 0
    for i in range(len(B_diff)):
        if B_diff[i] < lower_b or B_diff[i] > upper_b:
            count += 1

    # print("Number of Delta E Values outside Standard Deviation =", count)
    std_b = len(B_diff) - count
    print("Percentage of B_difference Values within Standard Deviation:", (round((std_b / len(B_diff) * 100), 2)))
    print("")

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
    # plt.show()


def deltaE():
    mean_e = statistics.mean(Delta_E)
    stddev_e = statistics.stdev(Delta_E)
    print("Lowest Delta E Value:", min(Delta_E))
    print("Highest Delta E Value:", max(Delta_E))
    print("Mean of Delta E Values:", round(mean_e, 2))
    print("Standard Deviation of Delta E Values:", round(stddev_e, 2))

    lower_e = mean_e - stddev_e
    upper_e = mean_e + stddev_e
    count = 0
    for i in range(len(Delta_E)):
        if Delta_E[i] < lower_e or Delta_E[i] > upper_e:
            count += 1

    # print("Number of Delta E Values outside Standard Deviation =", count)
    std_w = len(Delta_E) - count
    print("Percentage of Delta E Values within Standard Deviation:", (round((std_w / len(Delta_E) * 100), 2)))


LAB_diff(L_diff, A_diff, B_diff)
