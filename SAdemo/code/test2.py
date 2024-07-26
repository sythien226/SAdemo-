# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import math
import torch.nn as nn
import torchvision
from torchvision import models, transforms   
import matplotlib.pyplot as plt
from PIL import Image

import sys
import os
import warnings

# Suppress torch.meshgrid warning
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# Add yolov5_face directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5_face'))

cudnn.benchmark = True

from yolov5_face.models.experimental import attempt_load
from yolov5_face.utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from yolov5_face.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5_face.utils.plots import plot_one_box
from yolov5_face.utils.torch_utils import select_device, load_classifier, time_synchronized

weights = './static/models/yolov5s.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 4, (img1_shape[0] - img0_shape[0] * gain) / 4
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]
    coords[:, :10] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    coords[:, 8].clamp_(0, img0_shape[1])
    coords[:, 9].clamp_(0, img0_shape[0])
    return coords

def distance(A, B):
    x1, y1 = A
    x2, y2 = B
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def crop_face(img, xyxy, conf, landmarks, class_num):
    h, w, c = img.shape
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cropped_face = img[y1:y2, x1:x2].copy()
    
    left_eye = (int(landmarks[0] - x1), int(landmarks[1] - y1))
    right_eye = (int(landmarks[2] - x1), int(landmarks[3] - y1))
    nose = (int(landmarks[4] - x1), int(landmarks[5] - y1))
    left_mouth = (int(landmarks[6] - x1), int(landmarks[7] - y1))
    right_mouth = (int(landmarks[8] - x1), int(landmarks[9] - y1))

    # Tính hệ số góc của đường thẳng nối hai mắt
    m = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
    
    # Tính điểm giao (x_B, y_B)
    # Phương trình của đường thẳng qua hai mắt: y = m * x + b
    b = left_eye[1] - m * left_eye[0]
    
    # Phương trình của đường thẳng vuông góc: y = -1/m * x + b'
    b_perp = nose[1] + (1/m) * nose[0]
    
    # Giải hệ phương trình để tìm x_B và y_B
    x_B = (b_perp - b) / (m + 1/m)
    y_B = m * x_B + b
    
    intersection_point = (int(x_B), int(y_B))

    # Vẽ điểm giaoB
    cv2.circle(cropped_face, intersection_point, 5, (255, 0, 0), -1)  # Màu đỏ

    
    
    # Tính điểm giao (x_E, y_E)
    # Phương trình của đường thẳng qua hai ben miệng: y = m * x + b1
    b1 = left_mouth[1] - m * left_mouth[0]
    
    # Phương trình của đường thẳng vuông góc: y = -1/m * x + b'
    b1_perp = nose[1] + (1/m) * nose[0]
    
    # Giải hệ phương trình để tìm x_E và y_E
    x_E = (b1_perp - b1) / (m + 1/m)
    y_E = m * x_E + b1
    
    intersection_point_E = (int(x_E), int(y_E))

    # Vẽ điểm giaoE
    cv2.circle(cropped_face, intersection_point_E, 5, (255, 0, 0), -1)  # Màu đỏ 

     
    # Vẽ đường nối hai miệng và đường vuông góc đi qua mũi
    cv2.line(cropped_face, left_mouth, right_mouth, (0, 255, 255), 2)  # Đường nối hai miệng (cyan)


    


    # Tính điểm giao C của trục ngang và đường thẳng vuông góc với trục ngang qua mũi
    intersection_point_C = (nose[0], 0)  # x = x của mũi, y = 0 (trục ngang trên cùng)
    
    # Vẽ điểm giao C
    cv2.circle(cropped_face, intersection_point_C, 5, (0, 0, 255), -1)  # Màu xanh dương
    
    # Vẽ các đường thẳng nối các điểm landmarks
    cv2.line(cropped_face, left_eye, right_eye, (255, 0, 0), 2)  # Đường thẳng giữa hai mắt
    cv2.line(cropped_face, right_eye, nose, (0, 255, 0), 2)      # Đường thẳng từ mắt phải đến mũi
    cv2.line(cropped_face, left_eye, nose, (0, 255, 0), 2)      # Đường thẳng từ mắt trái đến mũi
    cv2.line(cropped_face, intersection_point, nose, (0, 255, 0), 2) #đường thẳng từ mũi đến điểm giao
    cv2.line(cropped_face, intersection_point_C, nose, (0, 255, 0), 2)#đường thẳng nối từ mũi đến điểm giao C
    cv2.line(cropped_face, intersection_point_E, nose, (0, 255, 0), 2)#đường thẳng nối từ mũi đến điểm giao E
    cv2.circle(cropped_face, left_eye, 5, (0, 255, 0), -1) 
    cv2.circle(cropped_face, right_eye, 5, (0, 255, 0), -1) 
    cv2.circle(cropped_face, nose, 5, (0, 255, 0), -1)
    cv2.circle(cropped_face, left_mouth, 5, (0, 255, 0), -1) 
    cv2.circle(cropped_face, right_mouth, 5, (0, 255, 0), -1) 
    
    center_left = (left_eye[0] + left_mouth[0]) / 2, (left_eye[1] + left_mouth[1]) / 2
    center_right = (right_eye[0] + right_mouth[0]) / 2, (right_eye[1] + right_mouth[1]) / 2
    center_top = (left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2
    center_bottom = (left_mouth[0] + right_mouth[0]) / 2, (left_mouth[1] + right_mouth[1]) / 2

    distance1 = distance(nose,center_left)
    distance2 = distance(nose,center_right)
    distance3 = distance(nose,center_top)
    distance4 = distance(nose,center_bottom)
    distance5 = distance(nose,intersection_point)
    distance6 = distance(nose,intersection_point_E)
    distance7 = distance(nose,left_eye)
    distance8 = distance(nose,right_eye)
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    
    # Calculate vector from eye_center to nose
    vec_nose_to_eye = (-eye_center[0] + nose[0], -eye_center[1] + nose[1])
    
    # Angle with vertical (assuming vertical is along y-axis)
    angle_rad = math.atan2(vec_nose_to_eye[0], vec_nose_to_eye[1])
    angle_deg = math.degrees(angle_rad)

    
    left_center = ((left_eye[0] + left_mouth[0]) / 2, (left_eye[1] + left_mouth[1]) / 2)
    
    # Calculate vector from left_center to nose
    vec_center_to_nose = (nose[0] - left_center[0], nose[1] - left_center[1])
    
    # Angle with horizontal (assuming horizontal is along x-axis)
    angle_rad1 = math.atan2(vec_center_to_nose[1], vec_center_to_nose[0])
    angle_deg1 = math.degrees(angle_rad1)

    if angle_deg < 0:
        direction = "phải"
        angle_deg = -angle_deg
    else:
        direction = "trái"

    if angle_deg1 < 0:
        direction1 = "ngẩng"
        angle_deg1 = -angle_deg1
    else:
        direction1 = "cúi"    

    

    
    print(f"Angle between trục dọc and line from nose to eye_center: {angle_deg:.2f} độ, xoay {direction}")
    print(f"Angle between trục ngang and line from nose to left_center: {angle_deg1:.2f} độ, mặt {direction1}")
    print(left_eye,right_eye,nose ,left_mouth ,right_mouth)
    print(distance1 , distance2, distance3, distance4)
    print(f"khảng cách từ mũi đến điểm B:{distance5},khoảng cách từ mũi đến điểm E:{distance6}")
    print(f"khảng cách từ mũi đến mắt trái :{distance7},khoảng cách từ mũi đến mắt phải:{distance8}")
    if  center_left[0] < nose[0] and nose[0] < center_right[0]:
        #if abs(distance1-distance2) <= 10:
        if 0.8 < distance1 / distance2 < 1.3:   
            print("THANG MAT")
        elif distance1 < distance2:
            print("QUAY PHAI")
        else:
            print("QUAY TRAI")
    else:
        if distance1 < distance2:
            print("QUAY PHAI")
        else:
            print("QUAY TRAI")

    if  center_top[1] < nose[1] and nose[1] < center_bottom[1]:
        #if abs(distance3-distance4) <= 10:
        if 1.1 < distance3 / distance4 < 2.0:
            pass
        elif distance3 < distance4:
            print("NGANG DAU")
        else:
            print("CUI DAU")
    else:
        if distance3 < distance4:
            print("NGANG DAU")
        else:
            print("CUI DAU")
    return cropped_face

def Cut_face(model, device, img):
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5

    im = np.array(img)
    img0 = copy.deepcopy(im)
    h0, w0 = im.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for _, det in enumerate(pred):
        im0 = im
        if len(pred[0]) == 1:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    im0 = crop_face(im0, xyxy, conf, landmarks, class_num)

    return len(pred[0]), im0

def GetFace(im):
    model = load_model(weights, device)
    num_face, image_face = Cut_face(model, device, im)
    return num_face, image_face

if __name__ == '__main__':
    try:
        im = Image.open("code/data/cui_ngang/8.jpg")
        num_face, image_face = GetFace(im)
        print(f'Number of faces detected: {num_face}')
        if num_face > 0:
            plt.imshow(cv2.cvtColor(image_face, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide axes
            plt.savefig('detected_facexoay2.png')  # Save the figure
            print("Saved the detected face as 'detected_face.png'")
    except Exception as e:
        print(f"An error occurred: {e}")
