import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from imageMasking import spine_mask


def scatter_plot(image):
    #new_image = image

    # Read image and print dimensions
    # image = spine_mask()
    # image = cv2.imread("images/urchin_cropped.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plt.imshow(image)
    # plt.show()

    # print('shape', image.shape)
    r, c = image.shape[:2]
    out_r = 100
    new_image = cv2.resize(image, (int(out_r * float(c) / r), out_r))

    # pixels = new_image.reshape((-1, 3))

    # print('pixels shape :', pixels.shape)
    # print('New shape :', new_image.shape)

    # plt.figure(figsize=(14, 10))
    # plt.axis("off")

    # plt.subplot(121)
    # plt.title('Image After Masking')
    # plt.imshow(image)

    # plt.subplot(122)
    # plt.title('Image with decreased pixels.')
    # plt.imshow(new_image)
    # plt.show()

    rgb_plot(new_image)
    hsv_plot(new_image)
    lab_plot(new_image)


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


def lab_plot(new_image):
    lab_new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_new_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = new_image.reshape((np.shape(new_image)[0] * np.shape(new_image)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(a.flatten(), b.flatten(), l.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("a (red-blue)")
    axis.set_ylabel("b (green-yellow)")
    axis.set_zlabel("L (lightness)")
    plt.title('LAB 3D Scatter Plot')
    plt.show()
