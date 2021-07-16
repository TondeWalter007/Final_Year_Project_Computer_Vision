# Final Year Project

## Aim
Determining the correlation between spine colour of sea urchins and their gonad quality and size using a software program that can reliably estimate the spine colour from images of sea urchins

## Background

Sea urchins are harvested for their gonads or roe, which are their genital glands found inside the urchin. An obstacle that hinders the consistent production of high-quality sea urchin gonads is that there is no way for producers to assess the quality and size of the gonads before harvesting. A pratical method to determine the harvest readiness of the individual urchins is required which does not involve sampling and killing them

## Implementation

### Input
The program takes in a single-colour image as an input using OpenCV. The image is then passed into the object detection algorithm so that the sea urchin can be isolated from the background in case of a busy background so that the rest of the program can only focus on the most important part of the image. 

[main.py](main.py) takes in the input and executes the entire program.

### Object Detection
YOLOv3 detection model was used to train the dataset to detect the sea urchin. An urchin detection weight file, configuration file, and name file were produced during the process. The program detects and isolates the sea urchin from the rest of the image. The sea urchin is then cropped out of the image and passed into the next subcomponent.

The urchin detection weight file can be found on my google drive: https://drive.google.com/file/d/1A9S8BOzPc5fMzhI3G6wOc1dArV41gs0G/view?usp=sharing

[urchinDetector.py](urchinDetector.py) contains the object detection algorithm.

### Image Segmentation
Gaussian blur was used to smooth out the image to reduce the noise in the image to get better results in the segmentation. For the segmentation, Otsu’s thresholding was used to separate the spines of the urchin from the urchin itself, as well as the remaining background.  

Due to the method used to capture the image of the urchin, the image consists of noise in the form of a flash from the camera used, the thresholding technique was unable to completely isolate the urchin spines from the background around the urchin. As only a few spines are needed to extract the colour, a simple cropping technique was used to remove the background around the urchin. Because of the circular shape of the sea urchin, and the object detection algorithm which isolated the urchin from the rest of the image, the urchin encompassed about 80% of the image. Therefore, a circular cropping was performed, with a radius that is 2/5 of the width of the isolated image. This captured majority of the sea urchin’s spines and masked the rest of the image outside that radius. Some of the noise that appear in the image after segmentation was removed using morphological transformations in OpenCV. 

[imageMasking.py](imageMasking.py) contains the image segmentation algorithm.

### Colour Extraction
K-Means algorithm was proposed as the method of extracting the colour of the spines from the image. The number of clusters that were chosen as the input for this algorithm was three, one for the background, and two for the colour of the spines. The output of this K-Means clustering algorithm was the RGB values of the centres of the predefined number of clusters. The color-extration algorithm put the three RGB values resulting from the K-Means algorithm into a bar graph in order to visualize the colours extracted and their respective percentage of occurrence in the image. The colour which corresponds to the spine tips was then determined and extracted.

[colorExtraction.py](colorExtraction.py) contains the colour extraction algorithm.

### Output
The extracted spine tips colour value was then converted into the LAB colour space as this was the most exact means of representing colour and it is device independent. The colour value was then displayed on screen with the spine tips colour as the background alongside the original input image


