from ultralytics import YOLO
from utils import resize_image
import torch
import cv2
import time
import os
from config import *
from inference_rom import get_inference_rom

def configure(parser_rom):
    parser_rom.add_argument(
        "-s", "--source", type=str, required=True, help="path to video/cam/RTSP"
    )
    parser_rom.add_argument("--save", action="store_true", help="Save video")
    parser_rom.add_argument(
        "--hide", action="store_false", help="to hide inference window"
    )
    parser_rom.add_argument(
        "-p",
        "--pose",
        type=str,
        choices=[
            "yolov8n-pose",
            "yolov8s-pose",
            "yolov8m-pose",
            "yolov8l-pose",
            "yolov8x-pose",
            "yolov8x-pose-p6",
        ],
        default="yolov8n-pose",
        help="choose type of yolov8 pose model",
    )


def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = YOLO(f"{args.pose}.pt")
    model.to(device)
    # target_size = (600,600)
    if (
        args.source.endswith(".jpg")
        or args.source.endswith(".jpeg")
        or args.source.endswith(".png")
    ):
        img = cv2.imread(args.source)
        get_inference_rom(img, model)

        # save Image
        if args.save or args.hide is False:
            os.makedirs(os.path.join("runs", "detect"), exist_ok=True)
            path_save = os.path.join("runs", "detect", os.path.split(args.source)[1])
            cv2.imwrite(path_save, img)
            print(f"[INFO] Saved Image: {path_save}")

        # Hide video
        if args.hide:
            cv2.imshow("img", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
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
            os.makedirs(os.path.join("runs", "detect"), exist_ok=True)
            if not str(video_path).isnumeric():
                path_save = os.path.join("runs", "detect", os.path.split(video_path)[1])
            else:
                c = 0
                while True:
                    if not os.path.exists(
                        os.path.join("runs", "detect", f"cam{c}.mp4")
                    ):
                        path_save = os.path.join("runs", "detect", f"cam{c}.mp4")
                        break
                    else:
                        c += 1
            out_vid = cv2.VideoWriter(
                path_save,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (original_video_width, original_video_height),
            )

        p_time = 0

        while True:
            success, img = cap.read()
            if not success:
                print("[INFO] Failed to Read...")
                break

            # FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            print("FPS: ", fps)
            p_time = c_time
            # img = cv2.resize(img, target_size)
            get_inference_rom(img,model)

            if args.hide is False:
                frame_count += 1
                print(f"Frames Completed: {frame_count}/{length}")

            # Write Video
            if args.save or args.hide is False:
                out_vid.write(img)

            # Hide video
            if args.hide:
                cv2.imshow("img", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if args.save or args.hide is False:
            out_vid.release()
            print(f"[INFO] Output Video Saved in {path_save}")
        if args.hide:
            cv2.destroyAllWindows()
