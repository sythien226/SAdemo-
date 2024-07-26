# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import torch
from torch.autograd import Variable
from unet.models.unet_model import UNet_texture_front_ds  # chứa cấu trúc của model
from unet.dataset.preprocess import *                  # chứa phần định dạng dữ liệu
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Edit your path
path_model = './static/models/Wrinkle_seg.pth'
    
def change_color_binary_mask(mask_img, ori_img, color = [0, 255, 0]):
        mask = cv2.cvtColor(np.array(mask_img), cv2.COLOR_GRAY2BGR)
        mask = cv2.multiply(mask, np.array(color))        
        highlight_img = cv2.addWeighted(mask, 1, ori_img, 1, 0)
        return highlight_img

def draw_contours_small(mask_img, ori_img):
    # Tìm các đường viền trong hình ảnh
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_contours_img = ori_img
    # Vẽ các đường viền lên hình ảnh
    for contour in contours:
        # Kiểm tra diện tích của contour
        if cv2.contourArea(contour) < 25:  # xóa bỏ những vùng diện tích nhỏ hơn 25
            # Tìm Convex Hull cho contour
            hull = cv2.convexHull(contour)
            epsilon = 0.0001 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            draw_contours_img = cv2.drawContours(ori_img, [approx], 0, 0, 3)

    return draw_contours_img

def gen_texture(rgb_image):

    # Set Gaussian Kernel
    kernel1d = cv2.getGaussianKernel(21, 5)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    img_src = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    img_src = np.array(img_src, dtype=float)
    img_low = cv2.filter2D(img_src, -1, kernel2d)
    img_low = np.array(img_low, dtype=float)

    img_div = (img_src * 255.) / (img_low + 1.)
    img_div[img_div > 255.] = 255.
    img_div = np.array(img_div, dtype=np.uint8)
    img_div = 1 - img_div
    return img_div

def predict_wrinkle(origin_image, dlib_image, package_coordinate):

    image_crop = np.array(dlib_image)
    ori_image = np.array(origin_image)

    # ============================= Change params ======================================== 
    size = 640
    # gen texture
    img_texture = gen_texture(image_crop)
    # Image.fromarray(img_texture).save("debug/img_texture.png")

    # ================================ Paint eye, slip, nose ====================================
    mask_face = np.zeros_like(img_texture)
    mask_eye_right = np.zeros_like(img_texture)
    mask_eye_left = np.zeros_like(img_texture)
    mask_slip = np.zeros_like(img_texture)
    mask_nose = np.zeros_like(img_texture)

    cv2.drawContours(mask_face, package_coordinate['face'], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask_eye_right, package_coordinate['eye_right'], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask_eye_left, package_coordinate['eye_left'], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask_slip, package_coordinate['slip'], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ellipse = cv2.fitEllipse(np.array(package_coordinate['nose']))
    cv2.ellipse(mask_nose, ellipse, (255, 255, 255), -1, cv2.LINE_AA)

    # Áp dụng mask lên ảnh gốc để cắt vùng theo đa giác
    img_texture = cv2.bitwise_and(img_texture, mask_face) #+ ~mask_face
    img_texture = cv2.bitwise_and(img_texture, ~mask_eye_right) + mask_eye_right
    img_texture = cv2.bitwise_and(img_texture, ~mask_eye_left) + mask_eye_left
    img_texture = cv2.bitwise_and(img_texture, ~mask_slip) + mask_slip
    img_texture = cv2.bitwise_and(img_texture, ~mask_nose) + mask_nose
    # ===========================================================================================
    # Image.fromarray(img_texture).save("debug/img_texture_new.png")


    model = UNet_texture_front_ds(4, 2).to(device)
    model.load_state_dict(torch.load(path_model, map_location=device), strict=True)

    softmax_2d = torch.nn.Softmax2d()

    model.eval()
    EPS = 1e-12

    imgs, img_ttr = Dataset_Wrinkle_WDS_For_Pred(img_src=image_crop, img_ttr=img_texture, b_aug=False, max_pixel=255, height=size, width=size)

    with torch.no_grad():
        imgs = imgs.unsqueeze(0) 
        img_ttr = img_ttr.unsqueeze(0) 
        imgs = Variable(imgs).to(device)
        img_ttr = Variable(img_ttr).to(device)
        
        out_1, out_2, out_3, out_4 = model(imgs, img_ttr) # đầu vào model: ảnh và texture map

        out = torch.log(softmax_2d(out_1) + EPS)

        # =============== Save binary mask ======================
        transform = T.ToPILImage()
        one_img = transform(out.squeeze()[0]) # ảnh binary mask
        one_img = one_img.convert("L")
        one_img = one_img.point(lambda x: 0 if x < 1 else 255)

        # Đọc ảnh test gốc và lấy kích thước ảnh test gốc
        height, width, _ = image_crop.shape
        
        # Resize binary mask về kích thước ảnh test gốc
        one_img = cv2.resize(np.array(one_img), (width, height), interpolation=cv2.INTER_NEAREST)
        one_img = Image.fromarray(one_img)
        one_img = one_img.point(lambda x: 0 if x < 1 else 255)
        
        # Remove small area
        test_remove_small_area = np.array(one_img.copy())
        ori_remove_small_area = np.array(one_img.copy())
        test_remove = draw_contours_small(test_remove_small_area, ori_remove_small_area)

        # - Hightlight trên ảnh gốc
        highlight_img_ori = change_color_binary_mask(
                                                     cv2.resize(np.array(test_remove), (width, height), interpolation=cv2.INTER_NEAREST), 
                                                     ori_image
                                                     )

        return highlight_img_ori
        # highlight_img_ori = Image.fromarray(highlight_img_ori)
        # highlight_img_ori.save("highlight_img_ori.png")

