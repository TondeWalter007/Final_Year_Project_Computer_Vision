import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
from imageMasking import spine_mask
# from colorEqualization import CLAHE
import webcolors

from scipy.cluster.vq import kmeans, vq
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2


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


def least_dominant_color():
    # Read image and print dimensions
    image = spine_mask()
    # image2 = img.imread("images/spine.jpg")

    print('shape', image.shape)
    r, c = image.shape[:2]
    out_r = 500
    new_image = cv2.resize(image, (int(out_r * float(c) / r), out_r))

    pixels = new_image.reshape((-1, 3))

    print('pixels shape :', pixels.shape)
    print('New shape :', new_image.shape)

    plt.figure(figsize=(14, 10))
    plt.axis("off")

    plt.subplot(121)
    plt.title('Image After Masking')
    plt.imshow(image)

    plt.subplot(122)
    plt.title('Image with decreased pixels.')
    plt.imshow(new_image)
    plt.show()

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

    # Create a line plot of num_clusters and distortions
    plt.plot(num_clusters, distortions)
    plt.xticks(num_clusters)
    plt.title('Elbow Plot', size=18)
    plt.xlabel('Number of Clusters')
    plt.ylabel("Distortions")
    plt.show()

    # using sklearn's inbuilt kmean for clustering data and finding cluster centers i.e. means for clusters.
    n_clusters = 3
    k_means = KMeans(n_clusters)
    k_means.fit(pixels)
    k_colors = k_means.cluster_centers_
    labels = k_means.labels_
    print(k_colors)

    colors = np.asarray(k_means.cluster_centers_, dtype='uint8')
    print("Colors: ")
    print(colors)

    # print("Original Image --->")
    # plt.axis('off')
    # plt.imshow(image)
    # plt.show()

    # print("Dominant", 3, "Colours of Image --->")
    # plt.axis('off')
    # plt.imshow([colors])
    # plt.show()

    # percentage of each extracted colour in the image
    pixels_colourwise = np.unique(k_means.labels_, return_counts=True)[1]
    percentage = pixels_colourwise / pixels.shape[0]
    print("Percentage: ")
    print(percentage)

    plt.title('Dominance Of Colours', size=16)
    plt.bar(range(1, 4), percentage, color=np.array(colors) / 255)
    plt.ylabel('Percentage')
    plt.xlabel('Colours')
    plt.show()

    label_count = [0 for i in range(n_clusters)]
    for elements in labels:
        label_count[elements] += 1
    index_color = label_count.index(min(label_count))

    actual_name, closest_name = get_color_name(colors[index_color])
    dominant_rgb = colors[index_color]
    print("Least Dominant Color = Color of Spine Tips:")
    print("RGB:", dominant_rgb)
    print("Actual name -> " + str(actual_name) + ", Closest name -> " + closest_name)

    p = pixels.copy()
    for px in range(pixels.shape[0]):
        for _ in range(colors.shape[0]):
            p[px] = colors[k_means.labels_[px]]

            p[px] = colors[k_means.labels_[px]]
    img = p.reshape(out_r, -1, 3)

    plt.figure(figsize=(14, 10))

    plt.subplot(121)
    plt.title('Original Image with Decreased Pixels')
    plt.imshow(new_image)

    plt.subplot(122)
    plt.title('Regenerated Image using KMeans')
    plt.imshow(img)
    plt.show()


least_dominant_color()
