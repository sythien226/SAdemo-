from ultralytics import YOLO
from predict import *predict_yolov8_seg

# Load a model
model_path =  "./Pores.pt" # đường dẫn model
img_path = "G:/Skin O/Pore665/Pore665/test/cFair/368_Face_HQ.jpg"

# -----------------------------------------------------------------------------------------------------------------------------------
model = YOLO(model_path)  # load a custom model

predict_yolov8_seg(model, img_path, 'pores')
