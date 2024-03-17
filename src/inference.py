from utils import norm_kpts, plot_one_box, plot_skeleton_kpts
import pandas as pd
import cv2
import time

# Define global variables
sit_start_time = None
stand_start_time = None
sit_stand_transition_time = None
count = 0
first_sit_pose_detected = False
current_pose_state = None
sit_to_stand_start_time = None
timer = []

def get_inference(img, model, saved_model, class_names, col_names, conf, colors):
    global sit_start_time, stand_start_time, sit_stand_transition_time, count, first_sit_pose_detected, current_pose_state, sit_to_stand_start_time, timer
    results = model.predict(img)
    sit_stand_transition_times = []  # List to store sit_stand_transition_time for each iteration
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints.data):
            lm_list = []
            for pnt in pose:
                x, y = pnt[:2]
                lm_list.append([int(x), int(y)])

            if len(lm_list) == 17:
                pre_lm = norm_kpts(lm_list)
                if pre_lm is not None:  # Check if pre_lm is not None
                    data = pd.DataFrame([pre_lm], columns=col_names)
                    predict = saved_model.predict(data)[0]

                    if max(predict) > conf:
                        pose_class = class_names[predict.argmax()]

                        # Check if the pose is "Sit" or "Stand"
                        if pose_class in ["sit", "stand"]:
                            # Record start time for the corresponding pose
                            if pose_class == "sit":
                                if current_pose_state != "sit":
                                    sit_start_time = time.time()
                                    # Start the sit to stand timer only if the current pose is sit
                                    if not first_sit_pose_detected:
                                        sit_to_stand_start_time = time.time()
                                        first_sit_pose_detected = True
                            elif pose_class == "stand":
                                if current_pose_state != "stand":
                                    stand_start_time = time.time()

                            current_pose_state = pose_class

                            # Check for a transition from Stand to Sit
                            if current_pose_state == "stand" and first_sit_pose_detected:
                                sit_stand_transition_time = time.time() - sit_to_stand_start_time
                                text = f"Stand to Sit transition time: {sit_stand_transition_time/2.8:.2f} seconds"
                                cv2.putText(
                                    img,
                                    text,
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                sit_stand_transition_times.append(sit_stand_transition_time)
                                print(text)

                            print("predicted Pose Class: ", pose_class)

                            plot_one_box(
                                box.xyxy[0],
                                img,
                                colors[predict.argmax()],
                                f"{pose_class} {max(predict)}",
                            )
                            plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)

                else:
                    print("[INFO] Predictions are below the given Confidence!!")
    count += 1
    if sit_stand_transition_times:  # Check if the list is not empty
        last_transition_time = sit_stand_transition_times[-1]
        print("Last sit_stand_transition_time:", last_transition_time)
        return last_transition_time
    else:
        print("No transitions detected")
        return None
