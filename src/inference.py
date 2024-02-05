from utils import norm_kpts, plot_one_box, plot_skeleton_kpts
import pandas as pd
import cv2
import time

from config import *

def get_inference(img, model, saved_model, class_names, col_names, conf,colors,fps):
    global sit_start_time, stand_start_time, sit_stand_transition_time, count, first_sit_pose_detected, current_pose_state, sit_to_stand_start_time
    results = model.predict(img)
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints.data):
            lm_list = []
            for pnt in pose:
                x, y = pnt[:2]
                lm_list.append([int(x), int(y)])

            if len(lm_list) == 17:
                pre_lm = norm_kpts(lm_list)
                data = pd.DataFrame([pre_lm], columns=col_names)
                predict = saved_model.predict(data)[0]

                if max(predict) > conf:
                    pose_class = class_names[predict.argmax()]

                    if pose_class == 'Unknown Pose':
                        print('[INFO] Unknown Pose detected. Skipping time calculation...')
                    else:
                        # Check if the pose is "Sit" or "Stand"
                        if pose_class in ['sit', 'stand']:
                            # Record start time for the corresponding pose
                            if pose_class == 'sit':
                                sit_start_time = time.time()
                                sit_frame_number.append(count)
                                # Start the sit to stand timer only if the current pose is sit
                                if current_pose_state == 'sit' and not first_sit_pose_detected:
                                    sit_to_stand_start_time = time.time()
                                    first_sit_pose_detected = True
                                sit_frame_number.append(count)
                                # Start the sit to stand timer only if the current pose is sit
                                if current_pose_state == 'sit' and not first_sit_pose_detected:
                                    sit_to_stand_start_time = time.time()
                                    first_sit_pose_detected = True
                            elif pose_class == 'stand':
                                stand_start_time = time.time()
                                stand_frame_number.append(count)
                                stand_frame_number.append(count)

                            # Check for a transition from Sit to Stand or vice versa
                            if sit_start_time is not None and stand_start_time is not None:
                                sit_stand_transition_time = stand_start_time - sit_start_time
                                text = f"Sit to Stand transition time: {sit_stand_transition_time/fps:.2f} seconds"
                                if sit_stand_transition_time > 0:
                                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    print(text)

                        print('predicted Pose Class: ', pose_class)

                    plot_one_box(box.xyxy[0], img, colors[predict.argmax()], f'{pose_class} {max(predict)}')
                    plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)

                else:
                    print('[INFO] Predictions are below the given Confidence!!')
    count += 1
