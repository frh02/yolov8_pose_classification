from utils import norm_kpts, plot_one_box, plot_skeleton_kpts
import pandas as pd
import cv2

# def get_inference_sts(img, model, saved_model, class_names, col_names, conf, colors):
#   counter = 0
#   stage = None
#   results = model.predict(img)
#   for result in results:
    
#     for box, pose in zip(result.boxes, result.keypoints.data):
      
#       lm_list = []
#       for pnt in pose:
#         x, y = pnt[:2]
#         lm_list.append([int(x), int(y)])

#       if len(lm_list) == 17:
#         pre_lm = norm_kpts(lm_list)
#         data = pd.DataFrame([pre_lm], columns=col_names)
#         predict = saved_model.predict(data)[0]

#         if max(predict) > conf:
#           pose_class = class_names[predict.argmax()]

#           plot_one_box(box.xyxy[0], img, colors[predict.argmax()], f'{pose_class} {max(predict)}')
#           plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)
#           print(pose_class)
          
#           # Record start time for the corresponding pose
#           if pose_class == 'sit' and stage == None:
#             stage = "down"
#             print(stage)
#           elif pose_class == 'stand' and stage == "down":
#             stage = "up"
#             print(stage)
#           elif pose_class == 'sit' and stage == "up":
#             stage = None
#             counter += 1
#             print(stage)
#           cv2.putText(img, str(counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#   print(counter)


def get_inference_sts(img, model, saved_model, class_names, col_names, conf, colors):
    results = model.predict(img)
    counter = 0  # Initialize counter outside the loop
    stage = None  # Initialize stage outside the loop

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

                    plot_one_box(box.xyxy[0], img, colors[predict.argmax()], f'{pose_class} {max(predict)}')
                    plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)
                    print(pose_class)

                    # Record start time for the corresponding pose
                    if pose_class == 'sit' and stage is None:
                        stage = "down"
                        print(stage)
                    elif pose_class == 'stand' and stage == "down":
                        stage = "up"
                        print(stage)
                    elif pose_class == 'sit' and stage == "up":
                        counter += 1
                        stage = None  # Reset stage only after all conditions are checked
                        print(stage)

    cv2.putText(img, str(counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    print(counter)
