# MLEyeClassification
Using MediaPipe to extract facial features and then use Kmeans to classify and label individual facial features.

This was an independent study for school that I did, original goal was to classify multiple facial features and I may update or create future projects for that, but this just does the eyes, specifically the left eye.

There are four files:
- main.py
- faceNorm.py
- kmeansClust.py
- pipedClust.py
------
## main.py
This py file took the images in and used MediaPipe to extract the necessary datapoints from their landmark detection to get the 

![The facial detection from MediaPipe](https://raw.githubusercontent.com/bware22/MLEyeClassification/main/test_img10.png)

## faceNorm.py
This file normalized the data for each image in the dataset, using a normalization formula my professor gave me.

## kmeansClust.py
This was the file for generating the individual clusters, but it also plotted everything, as well as calculating the silhouette coefficient and the elbow curve. On top of converting the data from cartesian to polar coordinates.
![]()
![]()
![]()
## pipedClust.py
My attempts to clean up and streamline the process for kmeans, as well as doing Principal Component Analysis, but it was meaningless for the eyes I was working on.
