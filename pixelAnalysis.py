import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from urchinDetector import urchin_detector
from imageMasking import spine_mask
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, XYZColor
from colormath.color_objects import XYZColor, sRGBColor


def manual_pixels():
    r_m = [152, 161, 148, 136, 158, 173, 173, 163, 162, 176, 142, 135, 134, 129, 158, 124, 131, 159, 136, 176]
    g_m = [95, 101, 91, 93, 114, 134, 132, 103, 110, 126, 90, 92, 89, 84, 113, 78, 81, 113, 94, 129]
    b_m = [68, 67, 64, 74, 89, 117, 110, 69, 88, 103, 66, 73, 68, 61, 90, 62, 54, 87, 70, 109]

    #r_m = [153, 158, 146, 160, 156, 149, 132, 141, 139, 168, 147, 143, 154, 112, 150, 185, 119, 156, 161, 163]
    #g_m = [95, 101, 87, 101, 97, 93, 85, 100, 97, 106, 93, 85, 105, 75, 98, 126, 76, 107, 124, 125]
    #b_m = [73, 72, 71, 83, 65, 68, 59, 78, 75, 69, 67, 61, 75, 49, 85, 96, 60, 93, 98, 102]

    pixel_colors = []
    for i in range(len(r_m)):
        pixel_ele = [r_m[i]/255, g_m[i]/255, b_m[i]/255]
        pixel_colors.append(pixel_ele)


    print("MANUAL:")
    print("Length of R:", len(r_m))
    print("Length of G:", len(g_m))
    print("Length of B:", len(b_m))
    print("facecolors:", pixel_colors)

    l_list = []
    a_list = []
    b_list = []
    for i in range(len(r_m)):
        # print("Colour", i)
        # print("R:", r_m[i])
        # print("G:", g_m[i])
        # print("B:", b_m[i])

        rgb = sRGBColor(r_m[i] / 255, g_m[i] / 255, b_m[i] / 255)
        lab = convert_color(rgb, LabColor, through_rgb_type=XYZColor)
        # print("RGB:", rgb)
        # print("LAB:", lab)

        l = round(lab.lab_l, 2)
        a = round(lab.lab_a, 2)
        b = round(lab.lab_b, 2)

        l_list.append(l)
        a_list.append(a)
        b_list.append(b)

    #print("L_manual List:", l_list)
    #print("A_manual List:", a_list)
    #print("B_manual List:", b_list)
    #print("")

    return l_list, a_list, b_list, pixel_colors


def auto_pixels(r_list, g_list, b_list):
    r_a = r_list
    g_a = g_list
    b_a = b_list

    #print("Automatic:")
    #print("Length of R:", len(r_a))
    #print("Length of G:", len(g_a))
    #print("Length of B:", len(b_a))

    l_list_a = []
    a_list_a = []
    b_list_a = []
    for i in range(len(r_a)):
        # print("Colour", i)
        # print("R:", r_m[i])
        # print("G:", g_m[i])
        # print("B:", b_m[i])

        rgb = sRGBColor(r_a[i] / 255, g_a[i] / 255, b_a[i] / 255)
        lab = convert_color(rgb, LabColor, through_rgb_type=XYZColor)
        # print("RGB:", rgb)
        # print("LAB:", lab)

        l = round(lab.lab_l, 2)
        a = round(lab.lab_a, 2)
        b = round(lab.lab_b, 2)

        l_list_a.append(l)
        a_list_a.append(a)
        b_list_a.append(b)

    #print("L_auto List:", l_list_a)
    #print("A_auto List:", a_list_a)
    #print("B_auto List:", b_list_a)

    return l_list_a, a_list_a, b_list_a


def scatter_plot(image, l_list, a_list, b_list, pixel_colors):
    # r, c = image.shape[:2]
    # out_r = 500
    # img_resized = cv2.resize(image, (int(out_r * float(c) / r), out_r))

    # cv2.imshow('Original Image', img_resized)

    detected_urchin = urchin_detector(image)
    image_mask = spine_mask(detected_urchin)
    r, c = image_mask.shape[:2]
    out_r = 250
    new_image = cv2.resize(image_mask, (int(out_r * float(c) / r), out_r))

    pixels = new_image.reshape((-1, 3))

    print('pixels shape :', pixels.shape)
    print('New shape :', new_image.shape)

    height = new_image.shape[0]
    width = new_image.shape[1]

    r_list_auto = []
    g_list_auto = []
    b_list_auto = []

    for i in range(height):
        for j in range(width):
            pix_rgb = new_image[i, j]
            r_list_auto.append(pix_rgb[0])
            g_list_auto.append(pix_rgb[1])
            b_list_auto.append(pix_rgb[2])

    #print("r_auto_list:", r_list_auto)
    #print("g_auto_list:", g_list_auto)
    #print("b_auto_list:", b_list_auto)

    l_list_auto_, a_list_auto_, b_list_auto_ = auto_pixels(r_list_auto, g_list_auto, b_list_auto)

    #print("l_auto_list:", l_list_auto_)
    #print("a_auto_list:", a_list_auto_)
    #print("b_auto_list:", b_list_auto_)

    plt.figure(figsize=(14, 10))
    plt.axis("off")

    plt.subplot(121)
    plt.title('Image After Masking')
    plt.imshow(image_mask)

    plt.subplot(122)
    plt.title('Image with decreased pixels.')
    plt.imshow(new_image)
    plt.show()

    # rgb_plot(new_image)
    # hsv_plot(new_image)
    lab_plot(new_image, l_list, a_list, b_list, l_list_auto_, a_list_auto_,b_list_auto_, pixel_colors)


def rgb_plot(new_image):
    r, g, b = cv2.split(new_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = new_image.reshape((np.shape(new_image)[0] * np.shape(new_image)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.title('RGB 3D Scatter Plot')
    plt.show()


def hsv_plot(new_image):
    hsv_new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_new_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = new_image.reshape((np.shape(new_image)[0] * np.shape(new_image)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.title('HSV 3D Scatter Plot')
    plt.show()


def lab_plot(new_image, l_list, a_list, b_list, l_list_auto, a_list_auto, b_list_auto, pixel_color_m):
    #lab_new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2LAB)
    #l, a, b = cv2.split(lab_new_image)

    image = new_image

    l_m = l_list
    a_m = a_list
    b_m = b_list

    pixel_colors = image.reshape(-1, 3) / 255.0

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(a_list_auto, b_list_auto, l_list_auto, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("a (red-blue)")
    axis.set_ylabel("b (green-yellow)")
    axis.set_zlabel("L (lightness)")
    plt.title('LAB 3D Plot: Pixels of Segmented Image from Program')
    plt.xlim([0, 25])
    plt.ylim([-5, 35])
    axis.set_zlim(0, 100)
    plt.show()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(a_m, b_m, l_m, facecolors=pixel_color_m, marker="^")
    axis.set_xlabel("a (red-blue)")
    axis.set_ylabel("b (green-yellow)")
    axis.set_zlabel("L (lightness)")
    plt.title('LAB 3D Plot: Pixels of Manually Extracted Spine Tips')
    plt.xlim([0, 25])
    plt.ylim([-5, 35])
    axis.set_zlim(0, 100)
    plt.show()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    #pixel_colors = new_image.reshape((np.shape(new_image)[0] * np.shape(new_image)[1], 3))
    #norm = colors.Normalize(vmin=-1., vmax=1.)
    #norm.autoscale(pixel_colors)
    #pixel_colors = norm(pixel_colors).tolist()
    # print("Print:", pixel_colors)


    #a.flatten()
    #b.flatten()
    #l.flatten()
    axis.scatter(a_list_auto,b_list_auto ,l_list_auto , facecolors=pixel_colors, marker=".")
    axis.scatter(a_m, b_m, l_m, facecolors=pixel_color_m, marker="^")
    axis.set_xlabel("a (red-blue)")
    axis.set_ylabel("b (green-yellow)")
    axis.set_zlabel("L (lightness)")
    plt.title('LAB 3D Plot: Combined Pixels')
    plt.xlim([0, 25])
    plt.ylim([-5, 35])
    axis.set_zlim(0, 100)
    plt.show()


l_list, a_list, b_list, pixel_colors = manual_pixels()
img = cv2.imread('full_images/41A2.jpg', cv2.IMREAD_COLOR)
print("Pixel RGB:", img[1000, 800])
scatter_plot(img, l_list, a_list, b_list, pixel_colors)
