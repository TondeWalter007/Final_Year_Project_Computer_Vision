import cv2
from urchinDetector import urchin_detector
from imageMasking import spine_mask
from colorExtraction import least_dominant_color

img = cv2.imread('full_images/83A2.jpg', cv2.IMREAD_COLOR)

r, c = img.shape[:2]
out_r = 500
img_resized = cv2.resize(img, (int(out_r * float(c) / r), out_r))

#cv2.imshow('Original Image', img_resized)

detected_urchin = urchin_detector(img)
image_mask = spine_mask(detected_urchin)
least_dominant_color(image_mask)
