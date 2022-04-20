import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

# Load in the normalized points from the pickle file.
with open('leftEyePrimeFile.pickle', 'rb') as handle:
    leftEyePrimePoints = pickle.load(handle)

# Set up variables for the dataframe
rows = len(leftEyePrimePoints)
# Flatten them for getting it into a vector.
col = len(leftEyePrimePoints[0].flatten())
# Initialize an empty matrix of that size.
leftEyePrimeList = np.zeros((rows, col))

# Transform the dict into the matrix.
for key in leftEyePrimePoints:
    for idx, point in enumerate(leftEyePrimePoints[key].flatten()):
        leftEyePrimeList[key][idx] = point

# Transform it into a pandas datafram, with 32 columns (each point is "flattened") so (X1,Y1), (X2, Y2)... is now [X1, Y1, X2, Y2,...]
dataframe1 = pd.DataFrame(leftEyePrimeList,
                          columns=['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5', 'X6', 'Y6', 'X7', 'Y7',
                                   'X8',
                                   'Y8', 'X9', 'Y9', 'X10', 'Y10', 'X11', 'Y11', 'X12', 'Y12', 'X13', 'Y13', 'X14',
                                   'Y14',
                                   'X15', 'Y15', 'X16', 'Y16'])

# Attempts to standardize the data to see if it made a difference.
# X_std = StandardScaler().fit_transform(dataframe1)
# X2_std = MinMaxScaler().fit_transform(dataframe1)
# X_Norm = normalize(dataframe1)

# This was my attempt at calculating Principal Component Analysis, but ultimately wasn't useful for this particualr project because
# I can't reduce the columns at all because the eyes overall shapes and size require all 16 points.

# Save the scaled and normed data to excel for easy viewing.
# with pd.ExcelWriter("Output.xlsx") as writer:
#     dataframe1.to_excel(writer, sheet_name="Before Scaling")
#     pd.DataFrame(X_std).to_excel(writer, sheet_name="After Standard Scaler")
#     pd.DataFrame(X2_std).to_excel(writer, sheet_name="After MinMax Scaler")
#     pd.DataFrame(X_Norm).to_excel(writer, sheet_name="After Normalization")
# Create a PCA instance: pca
# pca = PCA(n_components=32)
# principalComponents = pca.fit_transform(X_Norm)
# # Plot the explained variances
# features = range(pca.n_components)
# plt.bar(features, pca.explained_variance_ratio_, color="black")
# plt.xlabel("PCA Features")
# plt.ylabel("Variance %")
# plt.xticks(features)
# plt.title("PCA for Normalization")
# plt.show()
# # Save components to a DataFrame
# PCA_components = pd.DataFrame(principalComponents)

# Using SKLearn pipelines to make easier changes.
preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler())
    ]
)

# For Kmeans clustering I set the values to accordingly
clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                n_clusters=7,
                init="random",
                n_init=50,
                max_iter=500,
            )
        )
    ]
)

# Pipe for the overall pipeline.
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

# Fit the dataframe into the pipline
pipe.fit(dataframe1)
preprocessed_data = pipe["preprocessor"].transform(dataframe1)
predicted_labels = pipe["clusterer"]["kmeans"].labels_
print(silhouette_score(preprocessed_data, predicted_labels))
print(predicted_labels)





