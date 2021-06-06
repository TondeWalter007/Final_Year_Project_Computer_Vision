import cv2
import numpy as np
from matplotlib import pyplot as plt


def spine_mask(img_copy):
    img = img_copy.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    r, c = img.shape[:2]
    out_r = 500
    new_img = cv2.resize(img_copy, (int(out_r * float(c) / r), out_r))

    # plt.imshow(new_img)
    # plt.show()

    # Image equalization
    # img_copy = CLAHE(img_copy)

    # global thresholding
    # ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    # ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Bitwise
    equalize = cv2.equalizeHist(th3)
    bit_and = cv2.bitwise_and(img_copy, img_copy, mask=equalize)

    kernel1 = np.ones((7, 7), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(bit_and, kernel1, iterations=1)
    dilation = cv2.dilate(erosion, kernel2, iterations=1)
    # closing = cv2.morphologyEx(bit_and, cv2.MORPH_CLOSE, kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # opening = cv2.morphologyEx(bit_and, cv2.MORPH_OPEN, kernel, iterations=4)
    spine_mask = dilation

    # plot all the images and their histograms
    # images = [img, 0, th1,
    #         img_copy, 0, bit_and,
    #          blur, 0, th3]
    # titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
    #          'Original Noisy Image','Histogram',"Otsu's Thresholding",
    #          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

    # for i in range(3):
    #    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    #    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    #    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    #    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    # plt.show()

    # plt.imshow(bit_and)
    # plt.show()

    return spine_mask


def centre_circle_mask(img):
    img_copy = img.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    r, c = img_copy.shape[:2]
    out_r = 500
    new_img = img #cv2.resize(img, (int(out_r * float(c) / r), out_r))
    circle_img = new_img.copy()

    height, width, channels = new_img.shape

    # Center coordinates
    center_coordinates = (int(width / 2), int(height / 2))

    # Radius of circle
    radius = int((width - (width / 5)) / 2)

    # Red color in BGR
    color = (0, 0, 0)

    # Line thickness of -1 px
    thickness = -1

    # Using cv2.circle() method
    # Draw a circle of red color of thickness -1 px
    mask = np.zeros(circle_img.shape[:2], dtype="uint8")
    cv2.circle(mask, center_coordinates, radius, 255, -1)
    masked = cv2.bitwise_and(circle_img, circle_img, mask=mask)

    # plt.figure(figsize=(14, 10))
    # plt.axis("off")

    # plt.subplot(121)
    # plt.title('Original Image')
    # plt.imshow(new_img)

    # plt.subplot(122)
    # plt.title("Cropped Image")
    # plt.imshow(masked)
    # plt.show()

    return masked

# img_copy = cv2.imread('images/42B3.jpg', cv2.IMREAD_COLOR)
# spine_mask(img_copy)
