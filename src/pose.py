from ultralytics import YOLO
import cv2
from utils import plot_skeleton_kpts, plot_one_box


model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(2)


while True:
    success, img = cap.read()
    if not success:
        print("[INFO] Failed to Read...")
        break

    results = model.predict(img)
    for result in results:
        for box, pose in zip(result.boxes, result.keypoints):
            plot_one_box(box.xyxy[0], img, (255, 0, 255), f"person {box.conf[0]:.3}")
            plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
