# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

import torch.nn as nn
import torchvision
from torchvision import models, transforms   
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('./yolov5_face/')

cudnn.benchmark = True

from models.experimental import attempt_load
from yolov5_face.utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from yolov5_face.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5_face.utils.plots import plot_one_box
from yolov5_face.utils.torch_utils import select_device, load_classifier, time_synchronized

weights='./static/models/yolov5s.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def crop_face(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cropped_face = img[y1:y2, x1:x2]
    
    return cropped_face

def Cut_face(
    model,
    device,
    img,
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz=(640, 640)
        
    im = np.array(img)
    img0 = copy.deepcopy(im)
    h0, w0 = im.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert from w,h,c to c,w,h
    img = img.transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]
    
    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # Process detections
    for _, det in enumerate(pred):  # detections per image
        im0 = im
        if(len(pred[0]) == 1):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

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

# if __name__ == '__main__':
#     im = Image.open("G:/MBv2SA_Pigment/data/Pigment20image/A/3098_Face_HQ.jpg")
#     num_face, image_face =GetFace(im)
#     print(num_face)
