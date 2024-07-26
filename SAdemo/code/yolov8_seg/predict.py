from ultralytics import YOLO
import cv2
import torch
import torchvision.transforms as T
import numpy as np

model_path =  "./static/models/Pores_seg.pt"
model = YOLO(model_path)

def change_color_binary_mask(mask_img, ori_img, color = [0, 255, 0]):
        mask = cv2.cvtColor(np.array(mask_img), cv2.COLOR_GRAY2RGB)
        mask = cv2.multiply(mask, np.array(color))
        highlight_img = cv2.addWeighted(mask, 0.5, ori_img, 1, 0)
        return highlight_img

def draw_contours(mask_img, ori_img):
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_contours_img = ori_img.copy()
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        draw_contours_img = cv2.drawContours(ori_img, [approx], 0, (255, 255, 0), 2)
    
    return draw_contours_img

def predict_yolov8_seg(im, object_name):
    
    img = np.array(im)
    H, W, _ = img.shape
    results = model(img)
    
    if len(results[0]) == 0: 
        mask_empty = np.zeros((H, W), dtype=np.uint8)
        return img
    else:
        for result in results:
            # get array results
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (people in COCO)
            object_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            object_masks = masks[object_indices]
            # scale for visualizing results
            object_mask = torch.any(object_masks, dim=0).int() * 255
            # save to file
        
        object_mask = object_mask.cpu().numpy()
        transform = T.ToPILImage()
        object_mask = transform(object_mask)
        object_mask = object_mask.convert("L")

        if object_name == 'pores':                                           
            highlight_img_ori = draw_contours(
                                            cv2.resize(np.array(object_mask), (W, H), interpolation=cv2.INTER_NEAREST),
                                            img
                                            )
            return highlight_img_ori
        
        # elif object_name == 'wrinkle':
        #     highlight_img_ori = change_color_binary_mask(
        #                                     cv2.resize(np.array(object_mask), (W, H), interpolation=cv2.INTER_NEAREST),
        #                                     img
        #                                     )
        #     cv2.imwrite("highlight_img_ori.png", highlight_img_ori)
