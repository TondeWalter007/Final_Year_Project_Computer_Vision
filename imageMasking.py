import cv2
import numpy as np
from matplotlib import pyplot as plt


def spine_mask(img_copy):
    img = img_copy.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Bitwise masking
    equalize = cv2.equalizeHist(th3)
    bit_and = cv2.bitwise_and(img_copy, img_copy, mask=equalize)

    kernel1 = np.ones((7, 7), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(bit_and, kernel1, iterations=1)
    dilation = cv2.dilate(erosion, kernel2, iterations=1)
    spine_mask = dilation

    # plt.show()
    plt.imshow(spine_mask)
    plt.show()

    spine_mask = centre_circle_mask(spine_mask)

    plt.imshow(spine_mask)
    plt.title("Segmented Image")
    plt.xlabel("x_pixels")
    plt.ylabel("y_pixels")

    plt.show()

    return spine_mask


def centre_circle_mask(img):
    new_img = img
    circle_img = new_img.copy()

    height, width, channels = new_img.shape

    # Center coordinates
    center_coordinates = (int(width / 2), int(height / 2))

    # Radius of circle
    radius = int((width - (width / 5)) / 2)

    # Using cv2.circle() method
    # Draw a circle of red color of thickness -1 px
    mask = np.zeros(circle_img.shape[:2], dtype="uint8")
    cv2.circle(mask, center_coordinates, radius, 255, -1)
    masked = cv2.bitwise_and(circle_img, circle_img, mask=mask)

    return masked
