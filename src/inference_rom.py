from keypoints_detection import *  # noqa: F403
import cv2
import numpy as np
from utils import calculate_angle, plot_skeleton_kpts, draw_angle_display


def get_inference_rom(img, model):
    detector = DetectKeypoint()  # Instantiate the keypoint detector  # noqa: F405

    # Call the model to get the keypoint results
    results = model.predict(img)

    # Iterate through the results
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints.data):
            # Call the keypoint extractor to get x and y coordinates
            keypoint_data = detector.get_xy_keypoint(result)

            # Extract x and y coordinates for specific keypoints
            hip_right = (keypoint_data[22], keypoint_data[23])  # Right hip
            hip_left = (keypoint_data[24], keypoint_data[25])  # Left hip
            ankle_right = (keypoint_data[30], keypoint_data[31])  # Right ankle
            ankle_left = (keypoint_data[32], keypoint_data[33])  # Left ankle
            knee_right = (keypoint_data[28], keypoint_data[29])  # Right knee
            knee_left = (keypoint_data[26], keypoint_data[27])  # Left knee

            # Print or process the extracted coordinates as needed
            print("Hip Right:", hip_right)
            print("Hip Left:", hip_left)
            print("Ankle Right:", ankle_right)
            print("Ankle Left:", ankle_left)
            print("Knee Right:", knee_right)
            print("Knee Left:", knee_left)
            angle = calculate_angle(hip_left, knee_left, ankle_left)
            angle_r = calculate_angle(hip_right, knee_right, ankle_right)
            print("Angle for the right knee is:", round(angle_r, 1))
            print("Angle for the left knee is:", round(angle, 1))
            # angle
            plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)
            cv2.putText(
                img,
                str(round(angle, 1)),
                tuple(np.multiply(knee_left, [640, 480]).astype(int)),  # noqa: F405
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 153, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                img,
                str(round(angle_r, 1)),
                tuple(np.multiply(knee_right, [640, 480]).astype(int)),  # noqa: F405
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (150, 150, 0),
                2,
                cv2.LINE_AA,
            )

            # Inside your loop where you calculate and print the angles, add the following lines to draw the angle displays
            draw_angle_display(
                img, knee_left, angle, (255, 153, 255)
            )  # Draw the angle display for the left knee
            draw_angle_display(
                img, knee_right, angle_r, (150, 150, 0)
            )  # Draw the angle display for the right knee
