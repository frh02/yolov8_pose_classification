from utils import norm_kpts, plot_one_box, plot_skeleton_kpts
import pandas as pd
import cv2

def get_inference_sts(img, model, saved_model, class_names, col_names, conf, colors, counter_list, state):
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

                    plot_one_box(box.xyxy[0], img, colors[predict.argmax()], f'{pose_class} {max(predict)}')
                    plot_skeleton_kpts(img, pose, radius=5, line_thick=2, confi=0.5)
                    print(pose_class)

                    # State machine to track pose changes
                    if state == 'sit' and pose_class == 'stand':
                        state = 'transition'
                        print(state)
                    elif state == 'stand' and pose_class == 'sit':
                        state = 'transition'
                        print(state)
                    elif state == 'transition' and pose_class == 'sit':
                        state = 'sit'
                        counter_list[0] += 1
                        print(state)
                    elif state == 'transition' and pose_class == 'stand':
                        state = 'stand'
                        print(state)

    cv2.putText(img, str(counter_list[0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    print(counter_list[0])
    return state
