import math

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.spatial import procrustes

mp_face_mesh = mp.solutions.face_mesh


def angle_between_vectors(vector1, vector2):
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    scalar_multiple = np.dot(vector1, vector2)
    angle = np.arccos(scalar_multiple)
    angle = angle / math.pi * 180

    return angle


def perpendicular_normalized_vector(vector1, vector2):
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    vector_perpendicular = np.cross(vector1, vector2)
    vector_perpendicular = vector_perpendicular / np.linalg.norm(vector_perpendicular)
    return vector_perpendicular


def remove_vector_component(vector, component):
    component = component / np.linalg.norm(component)
    angled_vector = vector - (np.dot(vector, component)) * component
    return angled_vector


def array_has_signal(array):
    checked_array = array[-20:]

    max_difference = max(checked_array) - min(checked_array)
    max_greater_than_zero = max(checked_array) > 0
    min_less_than_zero = min(checked_array) < 0
    max_in_range = abs(max(checked_array)) > (0.75 * abs(min(checked_array)))
    min_in_range = abs(min(checked_array)) > (0.75 * abs(max(checked_array)))

    return (max_difference > 1) and max_greater_than_zero and min_less_than_zero and max_in_range and min_in_range


def filter_average(array):
    return sum(array) / len(array)


def filter_median(array):
    sorted_array = sorted(array)
    length = len(sorted_array)
    if length % 2 == 0:
        return (sorted_array[length // 2 - 1] + sorted_array[length // 2]) / 2
    else:
        return sorted_array[(length - 1) // 2]


def filter_bilateral(array):
    pass  # todo

def filter_gauss(array):
    return filter_average(scipy.ndimage.gaussian_filter1d(array, 0.5))


if __name__ == "__main__":
    # For webcam input:
    # cap = cv2.VideoCapture(0)

    # mentioning absolute path of the video
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S001_006.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S002_003.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S002_028.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S002_035.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S003_009.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S004_003.mp4"
    video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S008_027.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S008_031.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S008_037.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S008_048.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S009_001.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S009_003.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S009_028.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S009_044.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S010_025.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S012_011.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S016_001.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S016_017.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S019_055.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S020_029.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S021_082.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S022_022.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S022_044.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S023_026.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S023_038.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S026_034.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S027_028.mp4"
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\nods\\S031_021.mp4"

    #Ebben nincsen biztosan:
    # video_path = "C:\\Users\\X\\PycharmProjects\\BAUM\\BAUM1a\\s001\\S001_004.mp4"

    # creating a video capture object
    cap = cv2.VideoCapture(video_path)

    filter = filter_gauss
    # fig for 3d plot
    #fig = plt.figure(figsize=(5, 5))
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.view_init(elev=90, azim=90)

    fig2, (ax1, ax2, ax3) = plt.subplots(3)

    vertical_array_landmarks = [0 for x in range(0)]
    vertical_array_procrustes = [0 for x in range(0)]
    horizontal_array_landmarks = [0 for x in range(0)]

    filter_landmarks = [0 for x in range(15)]
    filter_procrustes = [0 for x in range(15)]

    bologatas_counter = 0
    vertical_absolute_position = 0
    prev_bologatas = False
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
                    break

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
                    prev_read = landmarks
                    first_landmarks = landmarks
                    first_read = False

                #mtx1, mtx2, disparity = procrustes(first_landmarks, landmarks)

                # print(landmarks, mtx2, disparity, "\n")
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
                ###############################     446:bal szem, 35:jobb szem

                for idx, l in enumerate(landmarks):
                    cv2.circle(image, (int(l[0]), int(l[1])), 1, (255, 0, 0), -1)
                    # image = cv2.putText(image, str(idx), (int(l[0]), int(l[1])), font, fontScale, color, thickness, cv2.LINE_AA)

                cv2.line(image, (int(landmarks[10][0]), int(landmarks[10][1])),
                         (int(landmarks[152][0]), int(landmarks[152][1])), (0, 0, 255), 2)
                cv2.line(image, (int(landmarks[446][0]), int(landmarks[446][1])),
                         (int(landmarks[226][0]), int(landmarks[226][1])), (0, 0, 255), 2)

                # vector_face = landmarks[10] - landmarks[152]
                # length = math.sqrt(
                #    vector_face[0] * vector_face[0] + vector_face[1] * vector_face[1] + vector_face[2] * vector_face[2])

                # vector_face /= length

                # vertical_vector = np.array([0, -1, 0])

                # scalar_multiple = sum(vector_face * vertical_vector)
                # angle = np.arccos(scalar_multiple)
                # angle = angle / math.pi * 180

                # print(angle)

                ###########################################################################
                vector_face_vertical = landmarks[10] - landmarks[152]
                vector_face_horizontal = landmarks[446] - landmarks[226]

                vector_face_perpendicular = perpendicular_normalized_vector(vector_face_horizontal,
                                                                            vector_face_vertical)

                prev_vector_face_vertical = prev_read[10] - prev_read[152]
                prev_vector_face_horizontal = prev_read[446] - prev_read[226]

                angled_vector = remove_vector_component(vector_face_perpendicular, prev_vector_face_horizontal)

                # cv2.line(image, (int(landmarks[168][0]), int(landmarks[168][1])),
                #         (int(landmarks[168][0] + vector_face_perpendicular[0] * 100), int(landmarks[168][1] + vector_face_perpendicular[1] * 100)), (0, 0, 255), 2)

                angle = angle_between_vectors(angled_vector, prev_vector_face_vertical)

                vertical_absolute_position += angle - 90
                # filter_landmarks.append(angle - 90)
                # filter_landmarks = filter_landmarks[1:]
                #
                # vertical_array_landmarks.append(filter(filter_landmarks))
                # vertical_array_landmarks = vertical_array_landmarks[1:]
                vertical_array_landmarks.append(angle - 90)
                # vertical_array_landmarks = vertical_array_landmarks[1:]

                if array_has_signal(vertical_array_landmarks):
                    if (prev_bologatas == False):
                        print("bologatas", bologatas_counter)
                        bologatas_counter += 1
                        # vertical_array_procrustes.append(1)
                        prev_bologatas = True

                # else:
                #     prev_bologatas = False
                #     vertical_array_procrustes.append(0)
                #
                # vertical_array_procrustes = vertical_array_procrustes[1:]

                ###########################################################################
                vector_face_vertical = landmarks[10] - landmarks[152]
                vector_face_horizontal = landmarks[446] - landmarks[226]

                vector_face_perpendicular = perpendicular_normalized_vector(vector_face_horizontal,
                                                                            vector_face_vertical)

                prev_vector_face_vertical = prev_read[10] - prev_read[152]
                prev_vector_face_horizontal = prev_read[446] - prev_read[226]

                angled_vector = remove_vector_component(vector_face_perpendicular, prev_vector_face_vertical)

                # cv2.line(image, (int(landmarks[168][0]), int(landmarks[168][1])),
                #         (int(landmarks[168][0] + vector_face_perpendicular[0] * 100), int(landmarks[168][1] + vector_face_perpendicular[1] * 100)), (0, 0, 255), 2)

                angle = angle_between_vectors(angled_vector, prev_vector_face_horizontal)

                # vertical_absolute_position += angle - 90
                # filter_landmarks.append(angle - 90)
                # filter_landmarks = filter_landmarks[1:]
                #
                # vertical_array_landmarks.append(filter(filter_landmarks))
                # vertical_array_landmarks = vertical_array_landmarks[1:]
                horizontal_array_landmarks.append(angle - 90)
                # vertical_array_landmarks = vertical_array_landmarks[1:]
                ##########################################################################################################

                # ax1.cla()
                # ax2.cla()
                # ax3.cla()

                # ax1.plot(range(len(vertical_array_landmarks)), scipy.ndimage.gaussian_filter1d(vertical_array_landmarks, 2), label="vertical")
                # ax2.plot(range(len(horizontal_array_landmarks)),scipy.ndimage.gaussian_filter1d(horizontal_array_landmarks, 2), label="horizontal")

                # ax2.plot(range(len(vertical_array_procrustes)), vertical_array_procrustes, label="horizontal")
                #
                # difference = [vertical_array_landmarks[i] - vertical_array_procrustes[i] for i in
                #               range(len(vertical_array_landmarks))]
                # ax3.plot(range(len(vertical_array_landmarks)), difference, label="difference")

                #plt.show()

                # cv2.imshow('mesh', image)

                #if cv2.waitKey(5) & 0xFF == ord('q'):
                #    break

                # plot 3d landmarks
                #ax.cla()
                #ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2])
                # ax.scatter(mtx2[:, 0], mtx2[:, 1], mtx2[:, 2])

                #plt.legend()
                # plt.pause(0.001)

                prev_read = landmarks

            except Exception as e:
                print(e)
                #cv2.imshow('mesh', image)

    cap.release()

    gauss_vertical = scipy.ndimage.gaussian_filter1d(vertical_array_landmarks, 4.5)
    ax1.plot(range(len(gauss_vertical)), gauss_vertical,label="relative")

    derivative = []
    for i in range(len(gauss_vertical) - 1):
        derivative.append(gauss_vertical[i + 1] - gauss_vertical[i])
    ax2.plot(range(len(derivative)), derivative, label="derivative")

    gauss_horizontal = scipy.ndimage.gaussian_filter1d(horizontal_array_landmarks, 3)

    #ax3.plot(range(len(gauss_horizontal)),scipy.ndimage.gaussian_filter1d(vertical_array_landmarks, 3), label="horizontal")
    #ax2.plot(range(len(vertical_array_landmarks)), horizontal_array_landmarks, label="horizontal")

    div = []
    for i in range(len(derivative)):
        div.append(gauss_vertical[i] / (abs(derivative[i]) + 0.1))

    ax3.plot(range(len(div)), div, label="relative/derivative")
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()
    # while(True):
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break