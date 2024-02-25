from keras.models import load_model
from keras.models import save_model
import numpy as np
import h5py
import cv2
import math
import random


# Plot Skeleton Keypoints
def plot_skeleton_kpts(im, kpts, radius=5, shape=(640, 640), confi=0.5, line_thick=2):
    pose_palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    skeleton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]

    limb_color = pose_palette[
        [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
    ]
    kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    ndim = kpts.shape[-1]
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]]
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < confi:
                    continue
            cv2.circle(
                im,
                (int(x_coord), int(y_coord)),
                radius,
                color_k,
                -1,
                lineType=cv2.LINE_AA,
            )

    for i, sk in enumerate(skeleton):
        pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
        pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
        if ndim == 3:
            conf1 = kpts[(sk[0] - 1), 2]
            conf2 = kpts[(sk[1] - 1), 2]
            if conf1 < confi or conf2 < confi:
                continue
        if (
            pos1[0] % shape[1] == 0
            or pos1[1] % shape[0] == 0
            or pos1[0] < 0
            or pos1[1] < 0
        ):
            continue
        if (
            pos2[0] % shape[1] == 0
            or pos2[1] % shape[0] == 0
            or pos2[0] < 0
            or pos2[1] < 0
        ):
            continue
        cv2.line(
            im,
            pos1,
            pos2,
            [int(x) for x in limb_color[i]],
            thickness=line_thick,
            lineType=cv2.LINE_AA,
        )


def resize_image(img, target_size):
    """
    Resize image to target size while maintaining aspect ratio.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    if w / h > target_w / target_h:
        new_w = target_w
        new_h = int(h * (target_w / w))
    else:
        new_h = target_h
        new_w = int(w * (target_h / h))
    resized_img = cv2.resize(img, (new_w, new_h))
    return resized_img


# Normalize Keypoints
def norm_kpts(lm_list, torso_size_multiplier=2.5):
    max_distance = 0
    center_x = (lm_list[12][0] + lm_list[11][0]) * 0.5  # right_hip  # left_hip
    center_y = (lm_list[12][1] + lm_list[11][1]) * 0.5  # right_hip  # left_hip

    shoulders_x = (
        lm_list[6][0] + lm_list[5][0]  # right_shoulder
    ) * 0.5  # left_shoulder
    shoulders_y = (
        lm_list[6][1] + lm_list[5][1]  # right_shoulder
    ) * 0.5  # left_shoulder

    for lm in lm_list:
        distance = math.sqrt((lm[0] - center_x) ** 2 + (lm[1] - center_y) ** 2)
        if distance > max_distance:
            max_distance = distance
    torso_size = math.sqrt(
        (shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2
    )
    max_distance = max(torso_size * torso_size_multiplier, max_distance)

    pre_lm = list(
        np.array(
            [
                [
                    (landmark[0] - center_x) / max_distance,
                    (landmark[1] - center_y) / max_distance,
                ]
                for landmark in lm_list
            ]
        ).flatten()
    )

    return pre_lm


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


# Save and Load Keras Model with meta data
def load_model_ext(filepath):
    model = load_model(filepath, custom_objects=None, compile=False)
    f = h5py.File(filepath, mode="r")
    meta_data = None
    if "my_meta_data" in f.attrs:
        meta_data = f.attrs.get("my_meta_data")
    f.close()
    return model, meta_data


def save_model_ext(model, filepath, overwrite=True, meta_data=None):
    save_model(model, filepath, overwrite)
    if meta_data is not None:
        f = h5py.File(filepath, mode="a")
        f.attrs["my_meta_data"] = meta_data
        f.close()


def calculate_angle(A, B, C):
    # Calculate vectors AB and BC
    vector_AB = [B[0] - A[0], B[1] - A[1]]
    vector_BC = [C[0] - B[0], C[1] - B[1]]

    # Calculate dot product of AB and BC
    dot_product = vector_AB[0] * vector_BC[0] + vector_AB[1] * vector_BC[1]

    # Calculate magnitudes of vectors
    magnitude_AB = math.sqrt(vector_AB[0] ** 2 + vector_AB[1] ** 2)
    magnitude_BC = math.sqrt(vector_BC[0] ** 2 + vector_BC[1] ** 2)

    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude_AB * magnitude_BC)

    # Calculate angle in radians
    angle_rad = math.acos(cosine_angle)

    # Convert angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def draw_angle_display(img, knee_point, angle, color):
    center = tuple(
        np.multiply(knee_point, [640, 480]).astype(int)
    )  # Convert knee point to image coordinates
    radius = 50  # Radius of the semicircle
    thickness = 2  # Thickness of the arc
    start_angle = -90  # Start angle for the arc
    end_angle = start_angle + angle  # End angle based on the angle value

    # Draw the arc representing the angle
    cv2.ellipse(
        img,
        center,
        (radius, radius),
        0,  # Angle of rotation of the ellipse (not needed for a circle)
        start_angle,
        end_angle,
        color,
        thickness,
    )


def calculate_angle_new(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # angle = np.abs(radians*180.0/np.pi)
    # angle = np.abs(radians*180/np.pi)
    # angle =  angle % 360; 

    
    # angle = (angle + 360) % 360;  

     
    # if (angle < 180) :
    #     angle -= 180
        
    # return abs(angle)
    ang = np.degrees(radians)
    #####################
    if ang<0:
        ang=abs(180+ang)
        
    elif ang>0 and ang<180:
        ang=180-ang
    elif ang>=180:
        ang=abs(180-ang)
    ######################
    return ang# + 360 if ang < 0 else ang


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized