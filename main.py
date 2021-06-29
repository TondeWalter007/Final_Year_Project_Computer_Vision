import cv2
from urchinDetector import urchin_detector
from imageMasking import spine_mask
from colorExtraction import least_dominant_color

# sea urchin input image
img = cv2.imread('full_images/41A2.jpg', cv2.IMREAD_COLOR)

# Sea urchin detection algorithm
detected_urchin = urchin_detector(img)

# image segmentation algorithm
image_mask = spine_mask(detected_urchin)

# colour extraction algorithm
least_dominant_color(image_mask)
