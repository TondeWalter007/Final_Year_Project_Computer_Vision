import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import webcolors
from imageMasking import centre_circle_mask
from colorAnalysis import scatter_plot
import math

from scipy.cluster.vq import kmeans, vq
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
from colormath.color_objects import XYZColor, HSLColor, AdobeRGBColor
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, XYZColor
from colormath.color_objects import XYZColor, sRGBColor


# import warnings
# warnings.filterwarnings('ignore')


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(requested_color):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
        actual_name = None
    return actual_name, closest_name


def recognize_color(R, G, B):
    index = ["color", "color_name", "hex", "R", "G", "B"]
    csv = pd.read_csv('colors.csv', names=index, header=None)
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


def least_dominant_color(image_mask):
    # Read image and print dimensions
    image = centre_circle_mask(image_mask)
    # image2 = img.imread("images/spine.jpg")

    # print('shape', image.shape)
    r, c = image.shape[:2]
    out_r = 500
    new_image = image #cv2.resize(image, (int(out_r * float(c) / r), out_r))

    pixels = new_image.reshape((-1, 3))

    show_decreased_pixel_image(pixels, new_image, image)

    scatter_plot(new_image)

    # Create a line plot of num_clusters and distortions
    # line_plot_of_clusters(new_image)

    # using sklearn's inbuilt kmean for clustering data and finding cluster centers i.e. means for clusters.
    n_clusters = 3
    k_means = KMeans(n_clusters)
    k_means.fit(pixels)
    k_colors = k_means.cluster_centers_
    labels = k_means.labels_
    # print(k_colors)

    colors = np.asarray(k_means.cluster_centers_, dtype='uint8')
    print("Three Most Common RGB Colors Extracted: ")
    print(colors)
    print("")

    # print("Original Image --->")
    # plt.axis('off')x
    # plt.imshow(image)
    # plt.show()

    # print("Dominant", 3, "Colours of Image --->")
    # plt.axis('off')
    # plt.imshow([colors])
    # plt.show()

    # percentage of each extracted colour in the image
    pixels_colourwise = np.unique(k_means.labels_, return_counts=True)[1]
    percentage = pixels_colourwise / pixels.shape[0]
    print("Percentage of Each Color (In Order): ")
    print(percentage)
    print("")

    color_bar_plot(n_clusters, percentage, colors)

    label_count = [0 for i in range(n_clusters)]
    for elements in labels:
        label_count[elements] += 1
    index_color = label_count.index(min(label_count))

    # actual_name, closest_name = get_color_name(colors[index_color])
    dominant_rgb = colors[index_color]

    #print("Least Dominant Color of Spines:")
    #least_dominant(dominant_rgb)

    # index_color_1 = label_count.index(label_count[1])
    # index_color_2 = label_count.index(label_count[2])

    #print("Average Color of Spines:")
    #l_avg, a_avg, b_avg = average_color(colors)

    print("Extracted Color of Spines:")
    extracted_colours(colors)

    #show_regenerated_image(new_image, pixels, colors, k_means, out_r)

    #return l_avg, a_avg, b_avg


def show_decreased_pixel_image(pixels, new_image, image):
    # print('pixels shape :', pixels.shape)
    # print('New shape :', new_image.shape)
    # print("")

    plt.figure(figsize=(14, 10))
    plt.axis("off")

    plt.subplot(121)
    plt.title('Image After Masking')
    plt.imshow(image)

    plt.subplot(122)
    plt.title('Image with decreased pixels.')
    plt.imshow(new_image)
    plt.show()


def line_plot_of_clusters(new_image):
    r, g, b = [], [], []
    for row in new_image:
        for r_val, g_val, b_val in row:
            r.append(r_val)
            g.append(g_val)
            b.append(b_val)

    # using scipy's inbuilt scaler whiten to scale.
    scaled_red = whiten(r)
    scaled_blue = whiten(b)
    scaled_green = whiten(g)

    df = pd.DataFrame({'red': r, 'blue': b, 'green': g, 'scaled_red': scaled_red, 'scaled_blue': scaled_blue,
                       'scaled_green': scaled_green})
    df.head()

    distortions = []
    num_clusters = range(1, 10)

    for i in num_clusters:
        cluster_centers, distortion = kmeans(df[['scaled_red', 'scaled_blue', 'scaled_green']], i)
        distortions.append(distortion)

    plt.plot(num_clusters, distortions)
    plt.xticks(num_clusters)
    plt.title('Elbow Plot', size=18)
    plt.xlabel('Number of Clusters')
    plt.ylabel("Distortions")
    plt.show()


def least_dominant(dominant_rgb):
    d_rgb = sRGBColor(dominant_rgb[0] / 255, dominant_rgb[1] / 255, dominant_rgb[2] / 255)
    d_lab = convert_color(d_rgb, LabColor, through_rgb_type=XYZColor)
    print("Least Dominant Color = Color of Spine Tips:")
    print("RGB:", dominant_rgb)
    # print("LAB:", d_lab)
    # print("Actual name -> " + str(actual_name) + ", Closest name -> " + closest_name)
    # print("")

    l = round(d_lab.lab_l, 2)
    a = round(d_lab.lab_a, 2)
    b = round(d_lab.lab_b, 2)

    print("LAB: [" + str(l) + " " + str(a) + " " + str(b) + "]")
    print("")

    return l, a, b


def average_color(colors):
    color_copy = colors.copy()
    print("Colors:")
    print(color_copy)
    index = np.where(color_copy == [0, 0, 0])
    print("Index:", color_copy[index])
    removed_color = np.delete(color_copy, index)
    print("New Colors:")
    print(removed_color)

    # avg_dom = math.sqrt((colors[index_color_1][0] ** 2 + colors[index_color_2][0] ** 2) / 2), math.sqrt(
    # (colors[index_color_1][1] ** 2 + colors[index_color_2][1] ** 2) / 2), math.sqrt(
    # (colors[index_color_1][2] ** 2 + colors[index_color_2][2] ** 2) / 2)

    avg_dom = round(math.sqrt((removed_color[0] ** 2 + removed_color[3] ** 2) / 2), 2), round(math.sqrt(
        (removed_color[1] ** 2 + removed_color[4] ** 2) / 2), 2), round(math.sqrt(
        (removed_color[2] ** 2 + removed_color[5] ** 2) / 2), 2)

    color_name = recognize_color(avg_dom[0], avg_dom[1], avg_dom[2])

    avg_rgb = sRGBColor(avg_dom[0] / 255, avg_dom[1] / 255, avg_dom[2] / 255)
    avg_lab = convert_color(avg_rgb, LabColor, through_rgb_type=XYZColor)
    print("Average Dominant Color = Color of Spines:")
    print("Average RGB:", avg_dom)
    print("Color name:", color_name)

    l_avg = round(avg_lab.lab_l, 2)
    a_avg = round(avg_lab.lab_a, 2)
    b_avg = round(avg_lab.lab_b, 2)

    print("Average LAB: [" + str(l_avg) + " " + str(a_avg) + " " + str(b_avg) + "]")
    print("")

    #return l_avg, a_avg, b_avg


def show_regenerated_image(new_image, pixels, colors, k_means, out_r):
    p = pixels.copy()
    for px in range(pixels.shape[0]):
        for _ in range(colors.shape[0]):
            p[px] = colors[k_means.labels_[px]]

            p[px] = colors[k_means.labels_[px]]
    img_ = p.reshape(out_r, -1, 3)

    plt.figure(figsize=(14, 10))

    plt.subplot(121)
    plt.title('Original Image with Decreased Pixels')
    plt.imshow(new_image)

    plt.subplot(122)
    plt.title('Regenerated Image using KMeans')
    plt.imshow(img_)
    plt.show()


def color_bar_plot(n_clusters, percentage, colors):
    plt.title('Dominance Of Colours', size=16)
    plt.bar(range(1, n_clusters + 1), percentage, color=np.array(colors) / 255)
    plt.ylabel('Percentage')
    plt.xlabel('Colours')
    plt.show()


def extracted_colours(colors):
    min_l = 100
    for i in range(len(colors)):

        print("Colour", i)
        print("R:", colors[i][0])
        print("G:", colors[i][1])
        print("B:", colors[i][2])

        rgb = sRGBColor(colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255)
        lab = convert_color(rgb, LabColor, through_rgb_type=XYZColor)
        print("RGB:", rgb)
        print("LAB:", lab)

        l = round(lab.lab_l, 2)
        a = round(lab.lab_a, 2)
        b = round(lab.lab_b, 2)

        print("LAB: [" + str(l) + " " + str(a) + " " + str(b) + "]")
        print("")

        temp_l = l - 0

        if temp_l == 0:
            continue

        if temp_l < min_l:
            min_l = temp_l
            min_a = a
            min_b = b

    print("Minimum Lightness = Darker = Spine Tips")
    print("L: "+str(min_l)+" A: "+str(min_a)+" B: "+str(min_b))
    print("")

        #return l, a, b