import cv2
import numpy as np

print("URCHIN DETECTION START...")


def load_weights():
    """"Loads the Yolov3 trained hand detection model"""
    net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")
    classes = []
    with open("classes.names", "r") as f:
        classes = f.read().splitlines()  # extracts the names in the file

    return net, classes


def urchin_detection(urchin_img, net):
    """"Detects urchin from the given input"""
    blob = cv2.dnn.blobFromImage(urchin_img, 1 / 255, (416, 416), (0, 0, 0),
                                 swapRB=True, crop=False)  # Rescales, Resizes the image and swap BGR to RGB
    net.setInput(blob)  # Sets blob as the input to the model
    output_layers_names = net.getUnconnectedOutLayersNames()  # gets info from the trained model
    layers_outputs = net.forward(output_layers_names)  # makes the prediction of the urchin

    return layers_outputs


def bounding_boxes(height, width, layers_outputs):
    boxes = []
    confidences = []
    class_ids = []

    for output in layers_outputs:  # extracts info from layer_outputs
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)  # gets the highest score's location
            confidence = scores[class_id]  # extracts the highest score

            if confidence > 0.3:  # Threshold value of the confidence
                # centre x and y coordinates of detected hand bounding box
                centre_x = int(detection[0] * width)
                centre_y = int(detection[1] * height)
                # size (width and height) of the bounding box
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # location of the upper left corner of bounding box
                x = int(centre_x - w / 2)
                y = int(centre_y - h / 2)

                # append the acquired info to their variables
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # suppress redundant bounding boxes and keep highest score box
    bounding_box = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    return scores, boxes, confidences, bounding_box, class_ids


def urchin_isolation(bounding_box, boxes, classes, class_ids, confidences, img):
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    try:
        for i in bounding_box.flatten():
            x, y, w, h = boxes[i]  # extract the location and size of bounding box
            label = str(classes[class_ids[i]])  # extract the corresponding class (urchin)
            confidence = str(round(confidences[i], 2))  # extract the confidence of the detection
            # color = colors[i]
            # color = (0, 0, 255)
            # urchin_region = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # create the bounding box
            # urchin_region = cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1, (0, 0, 255), 2)

        urchin_box = img[y:(y + h), x:(x + w)]  # isolate the urchin region in the image input

    except:
        print("No Urchin Detected")
        exit()

    return urchin_box  # , urchin_region


def urchin_detector(img):
    # img = cv2.imread('images/test3.jpg')
    # img = cv2.imread('full_images/43A4.jpg')
    # r, c = img.shape[:2]
    # out_r = 500
    new_img = img  # cv2.resize(img, (int(out_r * float(c) / r), out_r))

    # cv2.imshow('Original Image', img)

    # height, width, number of channels in image
    height = new_img.shape[0]
    width = new_img.shape[1]

    net, classes = load_weights()
    layers_outputs = urchin_detection(new_img, net)

    scores, boxes, confidences, bounding_box, class_ids = bounding_boxes(height, width, layers_outputs)
    urchin_box = urchin_isolation(bounding_box, boxes, classes, class_ids, confidences, new_img)

    r_, c_ = urchin_box.shape[:2]
    out_r = 500
    urchin_box_resized = cv2.resize(urchin_box, (int(out_r * float(c_) / r_), out_r))

    # cv2.imshow('Original Image', img)
    # cv2.imshow('Urchin Image Bounding Box', new_img)
    cv2.imshow('Isolated Urchin', urchin_box_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return urchin_box

# testing()
