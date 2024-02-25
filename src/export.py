from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt') # Load the pose model 

model.export(format='tfjs')