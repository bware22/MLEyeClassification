import pickle
import numpy as np

# Retrieve the pickle file of face point data.
with open('leftEyeDictFile.pickle', 'rb') as handle:
    leftEyeDict = pickle.load(handle)

temp_radius = 0
# The dictionary for the prime points.
leftEyeDictPrime = {}

# Get everyone in the datasets radius, to calculate r_mu, which is for our normalization of the data.
for temp in leftEyeDict:
    temp_radius += leftEyeDict[temp][0]
# r_mu calculated.
r_mu = temp_radius / len(leftEyeDict)

# For loop to extract each set of points and then normalize them, and put them into another dictionary to maintain their spot in the dictionary.
# To break down the dictionary, each key maps to an image in the dataset, so key 0 is image 0 i.e. the first image in the dataset.
# The three items in each dictionary are the radius, then the pupil coordinates, and then the sixteen coordinates for determining the size of the eyes. 
for key in leftEyeDict:
    # Extract Radius from the dictionary element. First item is the radius
    l_radius = leftEyeDict[key][0]
    # Calculate the value for R mu over R.
    r_delta = r_mu / l_radius
    # Extract the pupil point. The second item in the dict is the pupil.
    # Could probably write:
    # x_zero, y_zero = leftEyeDict[key][1]
    # For more pythonic code.
    pupil = leftEyeDict[key][1]
    x_zero = pupil[0]
    y_zero = pupil[1]
    # The 16 eye coordinates are the 3rd item in the dict.
    eye_Coords = leftEyeDict[key][2]
    # List comprehension for doing the whole normalization formula at once and storing the 16 points back into the dict.
    points_Prime = np.array([np.multiply(np.subtract([point[0], point[1]], [x_zero, y_zero]), [r_delta, r_delta]) for point
                    in eye_Coords])
    # Now in our new dictionary, image 0 still maps to key 0, but its the normalized points.
    leftEyeDictPrime[key] = points_Prime

# Store it back into another pickle file for easier handling in the future.
with open('leftEyePrimeFile.pickle', 'wb') as handle:
    pickle.dump(leftEyeDictPrime, handle, protocol=pickle.HIGHEST_PROTOCOL)
