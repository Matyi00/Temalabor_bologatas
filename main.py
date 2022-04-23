import math

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

mp_face_mesh = mp.solutions.face_mesh



def angle_between_vectors(vector1, vector2):

    length1 = math.sqrt(sum([x**2 for x in vector1]))
    length2 = math.sqrt(sum([x**2 for x in vector2]))

    vector1 = [x / length1 for x in vector1]
    vector2 = [x / length2 for x in vector2]

    scalar_multiple_vector = []
    for i in range(len(vector1)):
        scalar_multiple_vector.append(vector1[i] * vector2[i])

    scalar_multiple = sum(scalar_multiple_vector)
    angle = np.arccos(scalar_multiple)
    angle = angle / math.pi * 180

    return angle


if __name__ == "__main__":
    # For webcam input:
    cap = cv2.VideoCapture(0)

    # fig for 3d plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=90, azim=90)

    fig2, (ax1, ax2, ax3) = plt.subplots(3)

    vertical_array = [0 for x in range(100)]
    horizontal_array = [0 for x in range(100)]


    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            # refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        first_read = True

        while cap.isOpened():
            try:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue


                shape_y, shape_x = image.shape[:2]
                landmark_scaling = np.array([shape_x, shape_y, shape_x])

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        landmarks = [[l.x, l.y, l.z] for l in face.landmark]

                landmarks = np.array(landmarks) * landmark_scaling

                if first_read:
                    first_landmarks = landmarks
                    first_read = False


                mtx1, mtx2, disparity = procrustes(first_landmarks, landmarks)


                #print(landmarks, mtx2, disparity, "\n")
                ###############################
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # org

                # fontScale
                fontScale = 0.2

                # Blue color in BGR
                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 1
                # plot 2d landmarks on image
                ###############################     10:homlok, 152:áll
                ###############################     446:homlok, 35:áll

                for idx, l in enumerate(landmarks):
                    cv2.circle(image, (int(l[0]), int(l[1])), 1, (255, 0, 0), -1)
                    #image = cv2.putText(image, str(idx), (int(l[0]), int(l[1])), font, fontScale, color, thickness, cv2.LINE_AA)


                cv2.line(image, (int(landmarks[10][0]), int(landmarks[10][1])),
                         (int(landmarks[152][0]), int(landmarks[152][1])), (0, 0, 255), 2)
                cv2.line(image, (int(landmarks[446][0]), int(landmarks[446][1])),
                         (int(landmarks[226][0]), int(landmarks[226][1])), (0, 0, 255), 2)

                #vector_face = landmarks[10] - landmarks[152]
                #length = math.sqrt(
                #    vector_face[0] * vector_face[0] + vector_face[1] * vector_face[1] + vector_face[2] * vector_face[2])

                #vector_face /= length

                #vertical_vector = np.array([0, -1, 0])

                #scalar_multiple = sum(vector_face * vertical_vector)
                #angle = np.arccos(scalar_multiple)
                #angle = angle / math.pi * 180

                #print(angle)

                ###########################################################################
                vector_face_vertical = landmarks[10] - landmarks[152]
                vector_face_vertical[0] = 0

                vertical_vector = np.array([0, -1, 0])

                difference = vector_face_vertical - vertical_vector
                if difference[2] >= 0:
                    sign = -1
                else:
                    sign = 1

                angle_vertical = angle_between_vectors(vector_face_vertical, vertical_vector)

                vertical_array.append(angle_vertical * sign)
                vertical_array = vertical_array[1:]


                #print(angle_vertical * sign)
                ###########################################################################
                #
                #
                #
                # vector_face_horizontal = landmarks[446] - landmarks[226]
                # vector_face_horizontal[1] = 0
                # length = math.sqrt(
                #     vector_face_horizontal[0] * vector_face_horizontal[0] + vector_face_horizontal[1] * vector_face_horizontal[1] +
                #     vector_face_horizontal[2] * vector_face_horizontal[2])
                #
                # vector_face_horizontal /= length
                #
                # horizontal_vector = np.array([1, 0, 0])
                #
                # difference = vector_face_horizontal - horizontal_vector
                # if difference[2] >= 0:
                #     sign = -1
                # else:
                #     sign = 1
                #
                # scalar_multiple = sum(vector_face_horizontal * horizontal_vector)
                # angle_horizontal = np.arccos(scalar_multiple)
                # angle_horizontal = angle_horizontal / math.pi * 180
                #
                # #print(angle_horizontal * sign)
                # #horizontal_array.append(angle_horizontal * sign)
                # horizontal_array.append(angle_horizontal * sign)
                # horizontal_array = horizontal_array[1:]
                ##############################################################################################
                vector_face_horizontal = landmarks[10] - landmarks[152]
                vector_face_horizontal[0] = 0
                length = math.sqrt(
                    vector_face_horizontal[0] * vector_face_horizontal[0] + vector_face_horizontal[1] *
                    vector_face_horizontal[1] +
                    vector_face_horizontal[2] * vector_face_horizontal[2])

                vector_face_horizontal /= length

                horizontal_vector = mtx2[10] - mtx2[152]
                length = math.sqrt(
                    horizontal_vector[0] * horizontal_vector[0] + horizontal_vector[1] *
                    horizontal_vector[1] +
                    horizontal_vector[2] * horizontal_vector[2])
                horizontal_vector /= length

                difference = vector_face_horizontal - horizontal_vector
                if difference[2] >= 0:
                    sign = -1
                else:
                    sign = 1

                scalar_multiple = sum(vector_face_horizontal * horizontal_vector)
                angle_horizontal = np.arccos(scalar_multiple)
                angle_horizontal = angle_horizontal / math.pi * 180

                # print(angle_horizontal * sign)
                # horizontal_array.append(angle_horizontal * sign)
                horizontal_array.append(angle_horizontal * sign)
                horizontal_array = horizontal_array[1:]
                ##########################################################################################################


                ax1.cla()
                ax2.cla()
                ax3.cla()

                ax1.plot(range(len(vertical_array)), vertical_array, label="vertical")
                ax2.plot(range(len(horizontal_array)), horizontal_array, label="horizontal")

                difference = [vertical_array[i] - horizontal_array[i] for i in range(len(horizontal_array))]
                ax3.plot(range(len(horizontal_array)), difference, label="difference")

                #plt.show()


                cv2.imshow('mesh', image)

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                 #   break

                # plot 3d landmarks
                ax.cla()
                ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2])
                #ax.scatter(mtx2[:, 0], mtx2[:, 1], mtx2[:, 2])

                plt.legend()
                plt.pause(0.1)

            except:
                cv2.imshow('mesh', image)


    cap.release()
