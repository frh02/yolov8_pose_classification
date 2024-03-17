import cv2
import os
import pandas as pd
from ultralytics import YOLO
from utils import load_model_ext
from config import *  # noqa: F403
from inference_sts_y import get_inference_sts
import json
import random

def run_sts(source, hide, save, model, conf, pose):
    model_y = YOLO(f"{pose}.pt")
    saved_model, meta_str = load_model_ext(model)
    class_names = json.loads(meta_str)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

    video_path = source
    if video_path.isnumeric():
        video_path = int(video_path)

    cap = cv2.VideoCapture(video_path)
    state = "sit"  # Initial state
    counter_list = [0]
    
    # Extract video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        success, img = cap.read()
        if not success:
            print("[INFO] Failed to Read...")
            break

        state = get_inference_sts(
            img,
            model_y,
            saved_model,
            class_names,
            col_names,  # noqa: F405
            conf,
            colors,
            counter_list,
            state,
        )

    cap.release()

    # Save results to CSV file
    results_df = pd.DataFrame({"Video Name": [video_name], "Counter": [counter_list[0]]})
    results_csv_path = os.path.join("results", "results.csv")
    if os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_csv_path, index=False)

    print(f"[INFO] Results saved for {video_name}")

def process_mp4_files(directory, hide=True, save=True, model='models\model-sit10000_stand13000.h5', conf=0.5, pose="yolov8s-pose"):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            filepath = os.path.join(directory, filename)
            run_sts(filepath, hide, save, model, conf, pose)

# Example usage:
if __name__ == "__main__":
    directory_path = "processed"
    process_mp4_files(directory_path)
