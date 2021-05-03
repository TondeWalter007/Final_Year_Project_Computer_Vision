import cv2
import numpy as np
from matplotlib import pyplot as plt
#from colorEqualization import CLAHE


def spine_mask():
    img_copy = cv2.imread('images/urchin_cropped.jpg', cv2.IMREAD_COLOR)
    img = img_copy.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    r, c = img.shape[:2]
    out_r = 500
    new_img = cv2.resize(img_copy, (int(out_r * float(c) / r), out_r))

    plt.imshow(new_img)
    plt.show()

    #Image equalization
    #img_copy = CLAHE(img_copy)

    # global thresholding
    #ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    #ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Bitwise
    equalize = cv2.equalizeHist(th3)
    bit_and = cv2.bitwise_and(img_copy, img_copy, mask=equalize)

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

    #plt.imshow(bit_and)
    #plt.show()

    return bit_and
