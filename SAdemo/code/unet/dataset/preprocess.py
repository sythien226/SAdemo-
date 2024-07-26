import albumentations
import torch
import torch.utils.data
import numpy as np
import cv2

# ==========================================================================================================
def Dataset_Wrinkle_WDS_For_Pred(img_src, img_ttr, b_aug=False, max_pixel=255, height=640, width=640):

    meanRGB = [0.5, 0.5, 0.5]
    stdRGB = [0.25, 0.25, 0.25]


    if b_aug:
        transforms = albumentations.Compose([
            albumentations.Resize(height=height, width=width),
            albumentations.HorizontalFlip(),
            albumentations.ColorJitter(),
        ])
    else:
        transforms = albumentations.Compose([
            albumentations.Resize(height=height, width=width)
        ])

    transforms_norm = albumentations.Compose([
            albumentations.Normalize(mean=meanRGB, std=stdRGB, max_pixel_value=max_pixel),
        ])

    transforms_gray = albumentations.Compose([
        albumentations.Normalize(mean=meanRGB[0], std=stdRGB[0], max_pixel_value=max_pixel),
    ])

    transforms = transforms(image=img_src, mask=img_ttr)
    img_src = transforms['image']
    img_ttr = transforms['mask']

    transforms_img = transforms_norm(image=img_src)
    transforms_ttr = transforms_gray(image=img_ttr)

    inp_src = transforms_img['image']
    inp_ttr = transforms_ttr['image']

    inp_src = np.transpose(inp_src, [2, 0, 1])
    inp_ttr = np.expand_dims(inp_ttr, 0)

    inp_src = inp_src.astype(np.float32)
    inp_ttr = inp_ttr.astype(np.float32)

    # convert numpy -> torch
    inp_src = torch.from_numpy(inp_src)
    inp_ttr = torch.from_numpy(inp_ttr)

    return inp_src, inp_ttr
