import cv2.ft
import glob
import pickle
import numpy as np
import mediapipe as mp

# Detect face landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# Left and Right Iris Landmark points
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
# Left and Right eye indices list for landmark points. For left eye, we start at the inner corner point, and then go
# along the bottom of the eye counter-clockwise, until we reach the point 263, which is the outer corner,
# then we continued counter-clockwise around the top until point 398 which is at the uppermost inner corner
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# For right eye, we start at the outer corner of the eye in point 33, go counter-clockwise along the bottom lid again
# until we reach point 133, which is the innermost corner to the right eye. Then we go back along the top lid until
# we reach point 246 which is the outermost upper point.
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Use the glob for a folder of multiple images
# IMAGE_FILES = glob.glob('C:/FOLDER_PATH/*.jpg')
# Use a single image here
IMAGE_FILES = ["C:/SINGLE_IMAGE/singleImage.jpg"]
# Empty dictionary to store the data from each image.
leftEyeDict = {}
# Setting the drawing variable globally for the MediaPipe drawing.
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# This sets the parameters of FaceMesh, which is the full facial detection model from mediapipe The static_image_mode
# is for pictures and not webcam/video The max_num_faces is for the maximum number of faces it should detect in a
# given image (only ever 1 in our case). Refine_landmarks true will give us more data points for the iris and lips,
# we only need the iris for finding the pupil min_detection_confidence is the minimum certainty that it will accept
# for a face, might have to up the number if faces that fail get past it. Other possible values to try, .6 and 1
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    # For loop to go through all the images in the folder, getting the index in the image file array and then working
    # on each individual image from there.
    for idx, file in enumerate(IMAGE_FILES):
        # Load the image from the folder.
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing, this is needed because mediapipe use the RGB channels over
        # BGR that other facial detection models use. (Like DLib)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Getting the height and width of the image, using the slice operator : in the shape function ignores the
        # color channel return, we just need the height and width for converting the landmark points.
        img_h, img_w = image.shape[:2]
        # Print and draw face mesh landmarks on the image.
        # This test sees if the image passed or failed the min_detection_confidence. If it failed just go to next image
        # If it succeeded then move onto processing
        # Copy the image over to a temp variable to make all the changes to as to not alter the original file.
        # Might have to change that to saving the images in a new folder so I can see the drawn alterations.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        # Since we only work with images of single faces at a time, we will constantly be overwriting the image that is
        # in index 0, so we will always have to call that one.
        face_landmarks = results.multi_face_landmarks[0]
        # This inner for loop is for drawing each landmark, again not totally needed because its just for showing
        # where the landmarks appear on the image, once the facial detection ran in line 38, we can just work on the
        # image from there. For loop is unnecessary because we're only ever looking at images
        # of one face, so the result will always be stored in results.multi_face_landmarks[0] and then overwritten on
        # the next loop.
        # for face_landmarks in results.multi_face_landmarks:
        # This line will most likely be commented, as it prints out all the landmarks for a single image
        # Which isn't necessary or really readable by us.
        # print('face_landmarks:', face_landmarks)
        # Code commented out because I don't need these lines drawn on the face. These do the entire face excluding
        # The eyebrows, eyes, lips and circle around the face itself (the edge)
        # Technically don't need any, but I want something to be seen.
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_tesselation_style())
        # This does the eyes, eyebrows, lips, and face edge
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        # This does just the iris', might switch to just keep this one, not sure yet. Again no drawing is actually
        # necessary, because the landmarks were already placed and calculated.
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_iris_connections_style())
        # This code is in the outer for loop because I don't believe it needs to be inside the secondary for loop?
        # Turns out the inner loop is completely unnecessary, change later maybe?
        # I'm just picking out individual points and extracting the x,y coordinates, so I can get the minEnclosingCircle
        # The min circle will return the center and radius of the iris, which is the pupil and the radius I need for
        # The calculations. Not sure if its more or less accurate than just using the left and right point of the iris

        # Get and convert the points of the left iris
        leftIrisLandmarksNorm = np.array(
            [np.multiply([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y], [img_w, img_h]).astype(int) for i
             in LEFT_IRIS])
        leftEyeLandmarksNorm = np.array(
            [np.multiply([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y], [img_w, img_h]).astype(int) for i
             in LEFT_EYE])
        # Get and convert the points of the right iris.
        rightIrisLandmarksNorm = np.array(
            [np.multiply([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y], [img_w, img_h]).astype(int) for i
             in RIGHT_IRIS])
        # print("Left Iris Landmarks Normalized: ", leftIrisLandmarksNorm)
        # print("Right Iris Landmarks Normalized: ", rightIrisLandmarksNorm)
        # Calculate the pupil and the radius for left and right eye
        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(leftIrisLandmarksNorm)
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(rightIrisLandmarksNorm)
        # Turn pupil into np array
        pupil_left = np.array([l_cx, l_cy], dtype=np.int32)
        pupil_right = np.array([r_cx, r_cy], dtype=np.int32)
        # Draw on the circle around the iris and the point on the pupil, again not necessary.
        cv2.circle(annotated_image, pupil_left, int(l_radius), (255, 0, 255), 2)
        cv2.circle(annotated_image, pupil_left, radius=1, color=(225, 0, 100), thickness=2)
        cv2.circle(annotated_image, pupil_right, int(r_radius), (255, 0, 255), 2)
        cv2.circle(annotated_image, pupil_right, radius=1, color=(225, 0, 100), thickness=2)
        # Save image to new folder
        cv2.imwrite('test_img1' + str(idx) + '.png', annotated_image)
        # Store radius of left eye in an array to later create radius mu.
        leftEyeDict[idx] = l_radius, pupil_left, leftEyeLandmarksNorm

print(leftEyeDict)
# Store the results in a pickle file for retrival later.
with open('leftEyeDictFile.pickle', 'wb') as handle:
     pickle.dump(leftEyeDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
