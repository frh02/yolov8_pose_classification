# Training Phase for TUG and STS 
Experimenting with Pose classification


## How to get started

1. Clone the repository: 
```
https://github.com/frh02/yolov8_pose_classification.git
```

### Install Dependencies:
**Recommended**
```
conda create -n pose python=3.9 -y
conda activate pose
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda==11.8 -c pytorch -c nvidia -y
pip3 install -r requirements.txt
```
**OR**
```
pip3 install -r requirements.txt
```
### 🎒 Prepare Dataset
**Dataset Structure:**
```
├── Dataset
│   ├── class1
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── class2
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```

## 🤖 Train
### 🗃️ Create Landmark Dataset for each Classes
Convert pose images into pose lankmark and save to an **CSV** file.
So that we can train with that.

<details>
  <summary>Args</summary>
  
  `-p`, `--pose`: choose yolov8 pose model <br>
  **Choices:** 
  `yolov8n-pose`, `yolov8s-pose`, `yolov8m-pose`, `yolov8l-pose`, `yolov8x-pose`, `yolov8x-pose-p6` <br>
  `-i`, `--data`: path to data Dir <br>
  `-o`, `--save`: path to save csv file, eg: dir/data.csv
  
</details>

**Example:**
```
python3 src/generate_csv.py --pose yolov8n-pose --data dataset/train_data --save data.csv
```

### 🚆 Create DeepLearinng Model to predict Human Pose
Create a keras model to predict human poses.

<details>
  <summary>Args</summary>
  
  `-i`, `--data`: path to data Dir
  
</details>

**Example:**
```
python3 src/train.py --data data.csv
```

## 📺 Inference
Inference your Pose model.
#### Support
- Image
- Video
- Camera
- RTSP

<details>
  <summary>Args</summary>
  
  `-p`, `--pose`: choose yolov8 pose model <br>
  **Choices:** 
  `yolov8n-pose`, `yolov8s-pose`, `yolov8m-pose`, `yolov8l-pose`, `yolov8x-pose`, `yolov8x-pose-p6` <br>
  `-m`, `--model`: path to saved keras model <br>
  `-s`, `--source`: video path/cam-id/RTSP <br>
  `-c`, `--conf`: model prediction confidence (0<conf<1) <br>
  `--save`: to save video <br>
  `--hide`: hide video window

</details>

**Example:**
```
python3 src/inference.py --pose yolov8n-pose --model /runs/train4/ckpt_best.pth --source /test/video.mp4 --conf 0.66           # video
                                                                             --source /test/sample.jpg --conf 0.5 --save    # Image save
                                                                             --source /test/video.mp4 --conf 0.75 --hide    # to save and hide video window
                                                                             --source 0 --conf 0.45                         # Camera
                                            
```
