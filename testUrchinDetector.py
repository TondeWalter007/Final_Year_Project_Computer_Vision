import cv2
from urchinDetector import *
import os
import time

program_start = time.time()

path = 'detector_test_input'
outPath = 'detector_test_output_results'
count = 1
for image_path in os.listdir(path):
    image_start = time.time()
    print("Image "+str(count)+":")

    # img = cv2.imread('full_images/42A3.jpg')
    input_path = os.path.join(path, image_path)
    img = cv2.imread(input_path)
    r, c = img.shape[:2]
    out_r = 500
    new_img = cv2.resize(img, (int(out_r * float(c) / r), out_r))

    # cv2.imshow('Original Image', img)

    # height, width, number of channels in image
    height = new_img.shape[0]
    width = new_img.shape[1]

    net, classes = load_weights()
    layers_outputs = urchin_detection(new_img, net)

    scores, boxes, confidences, bounding_box, class_ids = bounding_boxes(height, width, layers_outputs)
    urchin_region = urchin_isolation(bounding_box, boxes, classes, class_ids, confidences, new_img)

    full_path_detected = os.path.join(outPath, 'detected_' + image_path)
    #full_path_cropped = os.path.join(outPath, 'cropped_' + image_path)

    cv2.imwrite(full_path_detected, urchin_region)
    #cv2.imwrite(full_path_cropped, urchin_box)
    print("SAVED!!!")
    print("")

    count += 1
    image_end = time.time()
    print(f"Runtime of the image analysis: {round((image_end - image_start), 2)} seconds")
    print("")

print("URCHIN DETECTION END...")
print("")
program_end = time.time()
print(f"Runtime of the program: {round((program_end - program_start), 2)} seconds")

