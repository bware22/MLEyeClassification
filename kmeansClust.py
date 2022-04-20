import pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

# Open the pickle file to extract the data.
with open('leftEyePrimeFile.pickle', 'rb') as handle:
    leftEyePrimePoints = pickle.load(handle)


dict_eye_vecs = {}
# Making the matrix for converting the dict.
rows = len(leftEyePrimePoints)
col = len(leftEyePrimePoints[0].flatten())
leftEyePrimeList = np.zeros((rows, col))

# Function to convert the cartesian points into polar coordinates.
# Just uses the standard formula for conversion.
def cartesian_to_polar_2d(cartesian):
    x, y = cartesian[:, 0], cartesian[:, 1]
    radius = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    return np.column_stack((radius, angle))

# Give the 16x2 set of points, convert all the points to polar and then into a vector.
def eye_to_vec(eye_cartesian):
    #Convert an eye shape matrix to a flat vector
    eye_polar = cartesian_to_polar_2d(eye_cartesian)
    return eye_polar.reshape((-1))


# The dataset of eyes; shape is 596 x 16 x 2
for key in leftEyePrimePoints:
    eyes = leftEyePrimePoints[key]
    # The data set of "eye vectors"; shape is 596 x 32
    eye_vecs = np.asarray(eye_to_vec(eyes))
    # Have the whole eye vector into the key dictionary spot, so key 0 ties to image 0
    dict_eye_vecs[key] = eye_vecs
# Now turn the dictionary of Eye Vectors into a matrix.
for key in dict_eye_vecs:
    for idx, point in enumerate(dict_eye_vecs[key]):
        leftEyePrimeList[key][idx] = point

# If I wanted to keep the data as cartesian points uncomment this, although
# pipeClust does this better with the pipes.
# for key in leftEyePrimePoints:
#     for idx, point in enumerate(leftEyePrimePoints[key].flatten()):
#         leftEyePrimeList[key][idx] = point
# scaler = MinMaxScaler()
# scaled_features = scaler.fit_transform(leftEyePrimeList)
# dataframe1 = pd.DataFrame(scaled_features,
#                           columns=['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5', 'X6', 'Y6', 'X7', 'Y7',
#                                    'X8',
#                                    'Y8', 'X9', 'Y9', 'X10', 'Y10', 'X11', 'Y11', 'X12', 'Y12', 'X13', 'Y13', 'X14',
#                                    'Y14',
#                                    'X15', 'Y15', 'X16', 'Y16'])

# Data frame of the polar coordinates, in vectors.
dataframe1 = pd.DataFrame(leftEyePrimeList,
                          columns=['r1', 't1', 'r2', 't2', 'r3', 't3', 'r4', 't4', 'r5', 't5', 'r6', 't6', 'r7', 't7',
                                   'r8',
                                   't8', 'r9', 't9', 'r10', 't10', 'r11', 't11', 'r12', 't12', 'r13', 't13', 'r14',
                                   't14',
                                   'r15', 't15', 'r16', 't16'])

# Setting the arguments for kmeans here.
kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 16,
    "max_iter": 512,
}

# Initialize the class object
# 7 Clusters, and using the kwargs
kmeans = KMeans(n_clusters=7, **kmeans_kwargs)
df = kmeans.fit_transform(dataframe1)

# predict the labels of clusters.
# label = kmeans.fit_predict(leftEyePrimeList)
label = kmeans.fit_predict(dataframe1)
# centroids = kmeans.cluster_centers_

# Getting unique labels, 7 for each cluster. 0-6
u_labels = np.unique(label)

# I wanted to print out an excel file where each image is tied to the cluster its in.
df2 = dataframe1.assign(ClusterVal=kmeans.labels_)
# It makes it easier for humans to read, and allows me to look at the end results.
df2["ClusterVal"].to_excel("imageANDClust.xlsx", sheet_name="Image with ClustVal")

# Create a rainbow of color values for all 16 points for the eyes.
colors = cm.rainbow(np.linspace(0, 1, 16))
# Create a figure
fig = plt.figure()
# For all the unique labels (7), plot all the eyes in that cluster over themselves onto a polar plot to see the grouping.
for i in u_labels:
    # Create a number of subplots equal to the individual labels.
    ax = fig.add_subplot(2, 4, i + 1, projection='polar')
    # Very ineffiecent but go through the whole list of eyes, and if it matches the current label you're on, grab it.
    for idx in range(len(label)):
        # Where the check happens, the last column is the unique label.
        if i == df2.iloc[idx][-1]:
            # Since the overall location integrity is kept, the index is always the same in both df2 and dict_eye_vecs, un-vectorize the data for the radius.
            radius = dict_eye_vecs[idx][::2]
            # Un-vectorize the angle.
            theta = dict_eye_vecs[idx][1::2]
            # Assign the color of the current point.
            c = ax.scatter(theta, radius, c=colors, alpha=1.0)
    # Set the title for the plot, with current unique label (0-6), I should probably +1 to i here so the plots are labeled 1-7 instead.
    ax.set_title("Polar Plot for cluster %i" % i)
# Save the subplot figure.
fig.savefig(f'cluster_subplots.png', dpi=300)
# Show it.
plt.show()

# This for loop makes individual plots for each cluster if we don't want subplots.
for i in u_labels:
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='polar')

    for idx in range(len(label)):
        if i == df2.iloc[idx][-1]:
            radius = dict_eye_vecs[idx][::2]
            theta = dict_eye_vecs[idx][1::2]

            c = ax.scatter(theta, radius, c=colors, alpha=1.0)
    ax.set_title("Polar Plot for cluster %i" % i)

    fig.tight_layout()
    fig.savefig(f'cluster_{i}.png', dpi=300)

plt.show()

# This code is for calculating the silhouette coeff.
# Make an empty list for silhouette coeff.
silhouette_coeff = []
# Check the silhouette coeff for 3-16 clusters.
for k in range(3, 16):
    # Run kmeans
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(dataframe1)
    # Calc the silhouette coeff
    score = silhouette_score(dataframe1, kmeans.labels_)
    # Append the result to the list.
    silhouette_coeff.append(score)
# Plot the list.
plt.style.use("fivethirtyeight")
plt.plot(range(3, 16), silhouette_coeff)
plt.xticks(range(3, 16))
plt.xlabel("Num of Clusters")
plt.ylabel("SSE")
plt.title("Silhouette Coefficient")
plt.show()

# This code is for calculating the elbow curve.
sse = []
# Again for clusters 3-16
for k in range(3, 16):
    # Run kmeans
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(dataframe1)
    # Again append the result to a list.
    sse.append(kmeans.inertia_)
# Plot the result
plt.style.use("fivethirtyeight")
plt.plot(range(3, 16), sse)
plt.xticks(range(3, 16))
plt.xlabel("Num of Clusters")
plt.ylabel("SSE")
# Find the knee of the curve for optimal clusters
kl = KneeLocator(range(3, 16), sse, curve="convex", direction="decreasing")
bestClust = kl.elbow
plt.axvline(x=bestClust, color="green", label="Knee Location")
plt.title("Elbow Curve Plot")
plt.show()
