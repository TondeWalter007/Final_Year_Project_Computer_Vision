import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image
from colorAnalysis import scatter_plot

# Loading our image with a cv2.imread() function
img = cv2.imread("images/urchin_cropped.jpg", cv2.IMREAD_COLOR)
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print('shape', RGB_img.shape)
r, c = RGB_img.shape[:2]
out_r = 500
new_img = cv2.resize(RGB_img, (int(out_r * float(c) / r), out_r))

new_img_copy = new_img.copy()

pixels = new_img_copy.reshape((-1, 3))

print('pixels shape :', pixels.shape)
print('New shape :', new_img_copy.shape)
height, width, channels = new_img_copy.shape

plt.figure(figsize=(14, 10))
plt.axis("off")

plt.subplot(121)
plt.title('Original Image')
plt.imshow(RGB_img)

plt.subplot(122)
plt.title('Image with decreased pixels.')
plt.imshow(new_img_copy)
plt.show()

# To get the value of the pixel (x=50, y=50), we would use the following code
(r, g, b) = new_img_copy[50, 50]
print("Pixel at (50,50):", new_img_copy[50, 50])
print("Pixel at (50, 50) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Using indexing we modified a whole region rather than one pixel
# For the top-left corner of the image, we can rewrite
# the color channels in folowing way:
numPix = 50
new_img_copy[0:numPix, 0:numPix] = [0, 255, 0]
new_img_copy[0:numPix, (width - numPix):516] = [0, 255, 0]
new_img_copy[(height - numPix):500, 0:numPix] = [0, 255, 0]
new_img_copy[(height - numPix):500, (width - numPix):516] = [0, 255, 0]

# Displaying updated image
plt.imshow(new_img_copy)
plt.title('Pixel Color Change')
plt.show()

img_temp = new_img.copy()

tl = img_temp[0:numPix, 0:numPix]
tl[:, :, 0], tl[:, :, 1], tl[:, :, 2] = np.average(tl, axis=(0, 1))

tr = img_temp[0:numPix, (width - numPix):516]
tr[:, :, 0], tr[:, :, 1], tr[:, :, 2] = np.average(tr, axis=(0, 1))

bl = img_temp[(height - numPix):500, 0:numPix]
bl[:, :, 0], bl[:, :, 1], bl[:, :, 2] = np.average(bl, axis=(0, 1))

br = img_temp[(height - numPix):500, (width - numPix):516]
br[:, :, 0], br[:, :, 1], br[:, :, 2] = np.average(br, axis=(0, 1))

plt.figure(figsize=(14, 10))

plt.subplot(121)
plt.title('Original Image')
plt.imshow(new_img)

plt.subplot(122)
plt.title('Averaged Top Left')
plt.imshow(tl)
plt.show()

# To get the value of the pixel (x=50, y=50), we would use the following code
y = 25
x = 25
(r1, g1, b1) = tl[y, x]
print("Pixel of Top Left at (25, 25) - Red: {}, Green: {}, Blue: {}".format(r1, g1, b1))
(r2, g2, b2) = tr[y, x]
print("Pixel of Top Right at (25, 25) - Red: {}, Green: {}, Blue: {}".format(r2, g2, b2))
(r3, g3, b3) = bl[y, x]
print("Pixel of Bottom Left at (25, 25) - Red: {}, Green: {}, Blue: {}".format(r3, g3, b3))
(r4, g4, b4) = br[y, x]
print("Pixel of Bottom Right at (25, 25) - Red: {}, Green: {}, Blue: {}".format(r4, g4, b4))

r_avg = int(math.sqrt(((r1 ** 2) + (r2 ** 2) + (r3 ** 2) + (r4 ** 2)) / 4))
print("Average Red Value:", r_avg)
g_avg = int(math.sqrt(((g1 ** 2) + (g2 ** 2) + (g3 ** 2) + (g4 ** 2)) / 4))
print("Average Green Value:", g_avg)
b_avg = int(math.sqrt(((b1 ** 2) + (b2 ** 2) + (b3 ** 2) + (b4 ** 2)) / 4))
print("Average Blue Value:", b_avg)

bg_avg = np.array((r_avg, g_avg, b_avg))

print("RGB Array:", bg_avg)

# print("")
# print("Thresholding:")

# if (r_avg - r1 >= 50) & (g_avg - g1) >= 50 & (b_avg - b1) >= 50:
#    print("Object")

# else:
#    print("Background")

bg_img = Image.new(mode="RGB", size=(width, height), color=(r_avg, g_avg, b_avg))
bg_img.save("images/bg_img.jpg")
bg_img_arr = np.asarray(bg_img)
plt.imshow(bg_img)
plt.title('Average Background Color [BG_IMG]')
plt.show()


img_copy = new_img.copy()
bg_img_jpg = cv2.imread("images/bg_img.jpg", cv2.IMREAD_COLOR)
bg_img_jpg = cv2.cvtColor(bg_img_jpg, cv2.COLOR_BGR2RGB)

hsv1 = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
hsv0 = cv2.cvtColor(bg_img_jpg, cv2.COLOR_RGB2HSV)

plt.figure(figsize=(14, 10))

plt.subplot(121)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(hsv1, cv2.COLOR_HSV2RGB))

plt.subplot(122)
plt.title('Average Background Color [BG_IMG_ARR]')
plt.imshow(cv2.cvtColor(hsv0, cv2.COLOR_HSV2RGB))
plt.show()

h1, s1, v1 = hsv1[0, 50]
h0, s0, v0 = hsv0[int(height/2), int(width/2)]

dh = min(abs(h1 - h0), 360 - abs(h1 - h0)) / 180
ds = abs(s1 - s0)
dv = abs(v1 - v0) / 255

distance = math.sqrt((dh * dh) + (ds * ds) + (dv * dv))

print("Euclidean Distance:", distance)

for y in range(0, height):
    for x in range(0, width):
        h1, s1, v1 = hsv1[y, x]
        h0, s0, v0 = hsv0[int(height / 2), int(width / 2)]

        dh = min(abs(h1 - h0), 360 - abs(h1 - h0)) / 180
        ds = abs(s1 - s0)
        dv = abs(v1 - v0) / 255

        distance = int(math.sqrt((dh * dh) + (ds * ds) + (dv * dv)))

        if distance <= 14:
            img_copy[y, x] = np.asarray([0, 0, 0])
        else:
            img_copy[y, x] = img_copy[y, x]

plt.imshow(img_copy)
plt.title('Thresholding pixel by pixel')
plt.show()

scatter_plot(img_copy)
