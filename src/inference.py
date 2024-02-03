from ultralytics import YOLO
from utils import norm_kpts, plot_one_box, plot_skeleton_kpts, load_model_ext
import pandas as pd
import cv2
import time
import os
import argparse
from config import *
import json
import random

# def configure(parser_tug):
#     parser_tug.add_argument(
#         "-m",
#         "--model", 
#         type=str, 
#         required=True,
#         help="path to saved keras model"
#     )
#     parser_tug.add_argument(
#         "-c", 
#         "--conf", 
#         type=float, 
#         default=0.25,
#         help="path to saved keras model"
#         )
#     parser_tug.add_argument("-s", "--source", type=str, required=True,
#                             help="path to video/cam/RTSP")
#     parser_tug.add_argument("--save", action='store_true',
#                             help="Save video")
#     parser_tug.add_argument("--hide", action='store_false',
#                             help="to hide inference window")
#     parser_tug.add_argument(
#         "-t", 
#         "--threshold", 
#         type=float, 
#         default=0.5,
#         help="threshold for detecting person"
#         )
#     parser_tug.add_argument(
#         "-p",
#         "--pose", 
#         type=str,
#         choices=[
#         'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 
#         'yolov8l-pose', 'yolov8x-pose', 'yolov8x-pose-p6'
#         ],
#         default='yolov8n-pose',
#         help="choose type of yolov8 pose model"
#         )
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--pose", type=str,
#                 choices=[
#                     'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 
#                     'yolov8l-pose', 'yolov8x-pose', 'yolov8x-pose-p6'
#                 ],
#                 default='yolov8n-pose',
#                 help="choose type of yolov8 pose model")
# ap.add_argument("-m", "--model", type=str, required=True,
#                 help="path to saved keras model")
# ap.add_argument("-c", "--conf", type=float, default=0.25,
#                 help="path to saved keras model")
# ap.add_argument("-s", "--source", type=str, required=True,
#                 help="path to video/cam/RTSP")
# ap.add_argument("--save", action='store_true',
#                 help="Save video")
# ap.add_argument("--hide", action='store_false',
#                 help="to hide inference window")
    


# def run(args):
#     # Keras pose model
#     saved_model, meta_str = load_model_ext(args['model'])
#     class_names = json.loads(meta_str)
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

#     if args['source'].endswith('.jpg') or args['source'].endswith('.jpeg') or args['source'].endswith('.png'):
#         img = cv2.imread(args['source'])
#         get_inference(img)

#         # save Image
#         if args['save'] or args['hide'] is False:
#             os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
#             path_save = os.path.join('runs', 'detect', os.path.split(args['source'])[1])
#             cv2.imwrite(path_save, img)
#             print(f"[INFO] Saved Image: {path_save}")
        
#         # Hide video
#         if args['hide']:
#             cv2.imshow('img', img)
#             if cv2.waitKey(0) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()

#     # Inference on Video/Cam/RTSP
#     else:
#         # Load video/cam/RTSP
#         video_path = args['source']
#         if video_path.isnumeric():
#             video_path = int(video_path)
#         cap = cv2.VideoCapture(video_path)

#         # Total Frame count
#         if args['hide'] is False:
#             length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0

#         # Write Video
#         if args['save'] or args['hide'] is False:
#             # Get the width and height of the video.
#             original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
            
#             # path to save videos
#             os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
#             if not str(video_path).isnumeric():
#                 path_save = os.path.join('runs', 'detect', os.path.split(video_path)[1])
#             else:
#                 c = 0
#                 while True:
#                     if not os.path.exists(os.path.join('runs', 'detect', f'cam{c}.mp4')):
#                         path_save = os.path.join('runs', 'detect', f'cam{c}.mp4')
#                         break
#                     else:
#                         c += 1
#             out_vid = cv2.VideoWriter(path_save, 
#                                 cv2.VideoWriter_fourcc(*'mp4v'),
#                                 fps, (original_video_width, original_video_height))

#         p_time = 0
#         while True:
#             success, img = cap.read()
#             if not success:
#                 print('[INFO] Failed to Read...')
#                 break

#             get_inference(img)
#             if args['hide'] is False:
#                 frame_count += 1
#                 print(f'Frames Completed: {frame_count}/{length}')

#             # FPS
#             c_time = time.time()
#             fps = 1/(c_time-p_time)
#             print('FPS: ', fps)
#             p_time = c_time

#             # Write Video
#             if args['save'] or args['hide'] is False:
#                 out_vid.write(img)

#             # Hide video
#             if args['hide']:
#                 cv2.imshow('img', img)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
            

#         cap.release()
#         if args['save'] or args['hide'] is False:
#             out_vid.release()
#             print(f"[INFO] Outout Video Saved in {path_save}")
#         if args['hide']:
#             cv2.destroyAllWindows()


# args = vars(ap.parse_args())

# col_names = [
#     '0_X', '0_Y', '1_X', '1_Y', '2_X', '2_Y', '3_X', '3_Y', '4_X', '4_Y', '5_X', '5_Y', 
#     '6_X', '6_Y', '7_X', '7_Y', '8_X', '8_Y', '9_X', '9_Y', '10_X', '10_Y', '11_X', '11_Y', 
#     '12_X', '12_Y', '13_X', '13_Y', '14_X', '14_Y', '15_X', '15_Y', '16_X', '16_Y'
# ]

# # YOLOv8 Pose Model
# model = YOLO(f"{args['pose']}.pt")

# # Add these variables to track time
# sit_start_time = None
# stand_start_time = None
# sit_stand_transition_time = None

# sit_frame_number = []
# stand_frame_number  = []

# first_sit_pose_detected = False
# count = 0 

# # Add a variable to track the current pose state
# current_pose_state = None

# # Add a variable to track the start time for sit to stand transition
# sit_to_stand_start_time = None


def get_inference(img, model, saved_model, class_names, col_names, conf,colors):
    global sit_start_time, stand_start_time, sit_stand_transition_time, count, first_sit_pose_detected
    global current_pose_state, sit_to_stand_start_time

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
                            elif pose_class == 'stand':
                                stand_start_time = time.time()
                                stand_frame_number.append(count)

                            # Check for a transition from Sit to Stand or vice versa
                            if sit_start_time is not None and stand_start_time is not None:
                                sit_stand_transition_time = stand_start_time - sit_start_time
                                text = f"Sit to Stand transition time: {sit_stand_transition_time:.2f} seconds"
                                if sit_stand_transition_time > 0:
                                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    print(text)

                        print('predicted Pose Class: ', pose_class)

                    plot_one_box(box.xyxy[0], img, colors[predict.argmax()], f'{pose_class} {max(predict)}')
                    plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)

                else:
                    print('[INFO] Predictions are below the given Confidence!!')
    count += 1

# # Keras pose model
# saved_model, meta_str = load_model_ext(args['model'])
# class_names = json.loads(meta_str)
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# # Inference Image
# if args['source'].endswith('.jpg') or args['source'].endswith('.jpeg') or args['source'].endswith('.png'):
#     img = cv2.imread(args['source'])
#     get_inference(img)

#     # save Image
#     if args['save'] or args['hide'] is False:
#         os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
#         path_save = os.path.join('runs', 'detect', os.path.split(args['source'])[1])
#         cv2.imwrite(path_save, img)
#         print(f"[INFO] Saved Image: {path_save}")
    
#     # Hide video
#     if args['hide']:
#         cv2.imshow('img', img)
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()

# # Inference on Video/Cam/RTSP
# else:
#     # Load video/cam/RTSP
#     video_path = args['source']
#     if video_path.isnumeric():
#         video_path = int(video_path)
#     cap = cv2.VideoCapture(video_path)

#     # Total Frame count
#     if args['hide'] is False:
#         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_count = 0

#     # Write Video
#     if args['save'] or args['hide'] is False:
#         # Get the width and height of the video.
#         original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
        
#         # path to save videos
#         os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
#         if not str(video_path).isnumeric():
#             path_save = os.path.join('runs', 'detect', os.path.split(video_path)[1])
#         else:
#             c = 0
#             while True:
#                 if not os.path.exists(os.path.join('runs', 'detect', f'cam{c}.mp4')):
#                     path_save = os.path.join('runs', 'detect', f'cam{c}.mp4')
#                     break
#                 else:
#                     c += 1
#         out_vid = cv2.VideoWriter(path_save, 
#                             cv2.VideoWriter_fourcc(*'mp4v'),
#                             fps, (original_video_width, original_video_height))

#     p_time = 0
#     while True:
#         success, img = cap.read()
#         if not success:
#             print('[INFO] Failed to Read...')
#             break

#         get_inference(img)
#         if args['hide'] is False:
#             frame_count += 1
#             print(f'Frames Completed: {frame_count}/{length}')

#         # FPS
#         c_time = time.time()
#         fps = 1/(c_time-p_time)
#         print('FPS: ', fps)
#         p_time = c_time

#         # Write Video
#         if args['save'] or args['hide'] is False:
#             out_vid.write(img)

#         # Hide video
#         if args['hide']:
#             cv2.imshow('img', img)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        

#     cap.release()
#     if args['save'] or args['hide'] is False:
#         out_vid.release()
#         print(f"[INFO] Outout Video Saved in {path_save}")
#     if args['hide']:
#         cv2.destroyAllWindows()
