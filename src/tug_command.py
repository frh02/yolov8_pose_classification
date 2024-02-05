from ultralytics import YOLO
from utils import load_model_ext
import cv2
import time
import os
from config import *
from inference import get_inference
import json
import random

def configure(parser_tug):
    parser_tug.add_argument(
        "-m",
        "--model", 
        type=str, 
        required=True,
        help="path to saved keras model"
    )
    parser_tug.add_argument(
        "-c", 
        "--conf", 
        type=float, 
        default=0.25,
        help="path to saved keras model"
        )
    parser_tug.add_argument("-s", "--source", type=str, required=True,
                            help="path to video/cam/RTSP")
    parser_tug.add_argument("--save", action='store_true',
                            help="Save video")
    parser_tug.add_argument("--hide", action='store_false',
                            help="to hide inference window")
    parser_tug.add_argument(
        "-t", 
        "--threshold", 
        type=float, 
        default=0.5,
        help="threshold for detecting person"
        )
    parser_tug.add_argument(
        "-p",
        "--pose", 
        type=str,
        choices=[
        'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 
        'yolov8l-pose', 'yolov8x-pose', 'yolov8x-pose-p6'
        ],
        default='yolov8n-pose',
        help="choose type of yolov8 pose model"
        )

def run(args):
    model = YOLO(f"{args.pose}.pt")

    # Keras pose model
    saved_model, meta_str = load_model_ext(args.model)
    class_names = json.loads(meta_str)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

    if args.source.endswith('.jpg') or args.source.endswith('.jpeg') or args.source.endswith('.png'):
        img = cv2.imread(args.source)
        get_inference(img,model,saved_model,class_names,col_names,args.conf,colors,fps)

        # save Image
        if args.save or args.hide is False:
            os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
            path_save = os.path.join('runs', 'detect', os.path.split(args.source)[1])
            cv2.imwrite(path_save, img)
            print(f"[INFO] Saved Image: {path_save}")
        
        # Hide video
        if args.hide:
            cv2.imshow('img', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

    # Inference on Video/Cam/RTSP
    else:
        # Load video/cam/RTSP
        video_path = args.source
        if video_path.isnumeric():
            video_path = int(video_path)
        cap = cv2.VideoCapture(video_path)

        # Total Frame count
        if args.hide is False:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

        # Write Video
        if args.save or args.hide is False:
            # Get the width and height of the video.
            original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # path to save videos
            os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
            if not str(video_path).isnumeric():
                path_save = os.path.join('runs', 'detect', os.path.split(video_path)[1])
            else:
                c = 0
                while True:
                    if not os.path.exists(os.path.join('runs', 'detect', f'cam{c}.mp4')):
                        path_save = os.path.join('runs', 'detect', f'cam{c}.mp4')
                        break
                    else:
                        c += 1
            out_vid = cv2.VideoWriter(path_save, 
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (original_video_width, original_video_height))

        p_time = 0
        while True:
            success, img = cap.read()
            if not success:
                print('[INFO] Failed to Read...')
                break

            # FPS
            c_time = time.time()
            fps = 1/(c_time-p_time)
            print('FPS: ', fps)
            p_time = c_time

            get_inference(img,model,saved_model,class_names,col_names,args.conf,colors,fps)
            if args.hide is False:
                frame_count += 1
                print(f'Frames Completed: {frame_count}/{length}')

            
            # Write Video
            if args.save or args.hide is False:
                out_vid.write(img)

            # Hide video
            if args.hide:
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            

        cap.release()
        if args.save or args.hide is False:
            out_vid.release()
            print(f"[INFO] Outout Video Saved in {path_save}")
        if args.hide:
            cv2.destroyAllWindows()
