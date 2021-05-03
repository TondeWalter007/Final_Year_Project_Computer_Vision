import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from imageMasking import spine_mask


def scatter_plot(image):
    # Read image and print dimensions
    # image = spine_mask()
    #image = cv2.imread("images/urchin_cropped.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #plt.imshow(image)
    #plt.show()

    print('shape', image.shape)
    r, c = image.shape[:2]
    out_r = 500
    new_image = cv2.resize(image, (int(out_r * float(c) / r), out_r))

    pixels = new_image.reshape((-1, 3))

    print('pixels shape :', pixels.shape)
    print('New shape :', new_image.shape)

    plt.figure(figsize=(14, 10))
    plt.axis("off")

    plt.subplot(121)
    plt.title('Image After Masking')
    plt.imshow(image)

    plt.subplot(122)
    plt.title('Image with decreased pixels.')
    plt.imshow(new_image)
    plt.show()

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

    hsv_new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_new_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.title('HSV 3D Scatter Plot')
    plt.show()

    hsv_min = (0,75,0)
    hsv_max = (25,225,200)

    #light_brown = (10, 100, 20)
    #dark_brown = (20, 255, 200)

    lo_square = np.full((10, 10, 3), hsv_min, dtype=np.uint8) / 255.0
    do_square = np.full((10, 10, 3), hsv_max, dtype=np.uint8) / 255.0

    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(do_square))
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_to_rgb(lo_square))
    plt.show()

    mask = cv2.inRange(hsv_new_image, hsv_min, hsv_max)
    result = cv2.bitwise_and(new_image, new_image, mask=mask)

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

    #light_white = (0, 0, 200)
    #dark_white = (145, 60, 255)

    #lw_square = np.full((10, 10, 3), light_white, dtype=np.uint8) / 255.0
    #dw_square = np.full((10, 10, 3), dark_white, dtype=np.uint8) / 255.0

    #plt.subplot(1, 2, 1)
    #plt.imshow(hsv_to_rgb(lw_square))
    #plt.subplot(1, 2, 2)
    #plt.imshow(hsv_to_rgb(dw_square))
    #plt.show()

    #mask_white = cv2.inRange(hsv_new_image, light_white, dark_white)
    #result_white = cv2.bitwise_and(new_image, new_image, mask=mask_white)

    #plt.subplot(1, 2, 1)
    #plt.imshow(mask_white, cmap="gray")
    #plt.subplot(1, 2, 2)
    #plt.imshow(result_white)
    #plt.show()

    #final_mask = mask + mask_white

    #final_result = cv2.bitwise_and(new_image, new_image, mask=final_mask)
    #plt.subplot(1, 2, 1)
    #plt.imshow(final_mask, cmap="gray")
    #plt.subplot(1, 2, 2)
    #plt.imshow(final_result)
    #plt.show()

