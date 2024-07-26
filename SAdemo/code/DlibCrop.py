import dlib
import numpy as np
import math
import cv2
from PIL import Image

detector = dlib.get_frontal_face_detector()
path_landmarks = "./static/models/shape_predictor_81_face_landmarks.dat"
predictor = dlib.shape_predictor(path_landmarks)

#tính toán điểm trung gian giữa 2 điểm A, B
def M2(pointA, pointB):
    a = round((pointA[0] + pointB[0])/2)
    b = round((pointA[1] + pointB[1])/2)
    return np.array((a, b))

#tính toán điểm trung gian giữa 3 điểm A, B, C
def M3(pointA, pointB, pointC):
    a = round((pointA[0] + pointB[0] + pointC[0])/3)
    b = round((pointA[1] + pointB[1] + pointC[1])/3)
    return np.array((a, b))
    
#crop smileline
def cropSmile(img, landmarks):
    smileline = []

    smileline.append(landmarks[4].tolist())
    smileline.append(landmarks[2].tolist())
    smileline.append(landmarks[29].tolist())
    smileline.append(landmarks[14].tolist())
    smileline.append(landmarks[12].tolist())
    smileline.append(landmarks[10].tolist())
    smileline.append(landmarks[9].tolist())
    smileline.append(landmarks[8].tolist())
    smileline.append(landmarks[7].tolist())
    smileline.append(landmarks[6].tolist())


    smileline = np.array(smileline)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [smileline], -1, (255, 255, 255), -1, cv2.LINE_AA)
    result = cv2.bitwise_and(img, mask)
    return result
          
#crop forehead
def cropForehead(img, landmarks):
    forehead = []

    for i in landmarks[17:26]:
            forehead.append(i.tolist())
    forehead.append(landmarks[78].tolist())
    forehead.append(landmarks[74].tolist())
    forehead.append(landmarks[79].tolist())
    forehead.append(landmarks[73].tolist())
    forehead.append(landmarks[72].tolist())
    forehead.append(landmarks[80].tolist())
    forehead.append(landmarks[71].tolist())
    forehead.append(landmarks[70].tolist())
    forehead.append(landmarks[69].tolist())
    forehead.append(landmarks[68].tolist())
    forehead.append(landmarks[76].tolist())
    forehead.append(landmarks[75].tolist())
    forehead.append(landmarks[77].tolist())
    
    
    forehead = np.array(forehead)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [forehead], -1, (255, 255, 255), -1, cv2.LINE_AA)
    result = cv2.bitwise_and(img, mask)
    return result
        
    #crop eye
def cropEye(img, landmarks):
    eye = []

    eye.append(landmarks[1].tolist())
    eye.append(landmarks[0].tolist())
    eye.append(M2(landmarks[36], landmarks[17]).tolist())
    eye.append(M2(landmarks[37], landmarks[18]).tolist())
    eye.append(M2(landmarks[38], landmarks[19]).tolist())
    eye.append(M2(landmarks[38], landmarks[20]).tolist())
    eye.append(M3(landmarks[21], landmarks[22], landmarks[27]).tolist())
    eye.append(M2(landmarks[42], landmarks[22]).tolist())
    eye.append(M2(landmarks[43], landmarks[23]).tolist())
    eye.append(M2(landmarks[44], landmarks[24]).tolist())
    eye.append(M2(landmarks[44], landmarks[25]).tolist())
    eye.append(M2(landmarks[45], landmarks[26]).tolist())
    eye.append(landmarks[16].tolist())
    eye.append(landmarks[15].tolist())
    eye.append(M2(landmarks[15], landmarks[30]).tolist())
    eye.append(M3(landmarks[29], landmarks[39], landmarks[42]).tolist())
    eye.append(M2(landmarks[1], landmarks[30]).tolist())
    
    eye = np.array(eye)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [eye], -1, (255, 255, 255), -1, cv2.LINE_AA)

    result = cv2.bitwise_and(img, mask)
    return result

def crop_face_remove_eye_slip_nose(img, landmarks, package_coordinate):
    
    face = []
    eye_left = []
    eye_right = []
    slip = []
    nose = []   

    if 'slip' not in package_coordinate:
      package_coordinate['slip'] = []
    if 'face' not in package_coordinate:
      package_coordinate['face'] = []
    if 'eye_left' not in package_coordinate:
      package_coordinate['eye_left'] = []
    if 'eye_right' not in package_coordinate:
      package_coordinate['eye_right'] = []
    if 'nose' not in package_coordinate:
      package_coordinate['nose'] = []
            
    # face
    for i in landmarks[0:16]:
            face.append(i.tolist())
    face.append(landmarks[78].tolist())
    face.append(landmarks[74].tolist())
    face.append(landmarks[79].tolist())
    face.append(landmarks[73].tolist())
    face.append(landmarks[72].tolist())
    face.append(landmarks[80].tolist())
    face.append(landmarks[71].tolist())
    face.append(landmarks[70].tolist())
    face.append(landmarks[69].tolist())
    face.append(landmarks[68].tolist())
    face.append(landmarks[76].tolist())
    face.append(landmarks[75].tolist())
    face.append(landmarks[77].tolist())

    # Slip
    slip.append(landmarks[48].tolist())
    slip.append(landmarks[49].tolist())
    slip.append(landmarks[50].tolist())
    slip.append(landmarks[51].tolist())
    slip.append(landmarks[52].tolist())
    slip.append(landmarks[53].tolist())
    slip.append(landmarks[54].tolist())
    slip.append(landmarks[55].tolist())
    slip.append(landmarks[56].tolist())
    slip.append(landmarks[57].tolist())
    slip.append(landmarks[58].tolist())
    slip.append(landmarks[59].tolist())
    slip.append(landmarks[60].tolist())
    
    # Eye left
    eye_left.append(landmarks[17].tolist())
    eye_left.append(landmarks[36].tolist())
    eye_left.append(landmarks[41].tolist())
    eye_left.append(landmarks[40].tolist())
    eye_left.append(landmarks[39].tolist())
    eye_left.append(landmarks[21].tolist())
    eye_left.append(landmarks[20].tolist())
    eye_left.append(landmarks[19].tolist())
    eye_left.append(landmarks[18].tolist())

    # Eye right
    eye_right.append(landmarks[22].tolist())
    eye_right.append(landmarks[42].tolist())
    eye_right.append(landmarks[47].tolist())
    eye_right.append(landmarks[46].tolist())
    eye_right.append(landmarks[45].tolist())
    eye_right.append(landmarks[26].tolist())
    eye_right.append(landmarks[25].tolist())
    eye_right.append(landmarks[24].tolist())
    eye_right.append(landmarks[23].tolist())

    # Nose
    nose.append(landmarks[29].tolist())
    nose.append(landmarks[31].tolist())
    nose.append(landmarks[32].tolist())
    nose.append(landmarks[33].tolist())
    nose.append(landmarks[34].tolist())
    nose.append(landmarks[35].tolist())


    face = np.array(face)
    slip = np.array(slip)
    eye_left = np.array(eye_left)
    eye_right = np.array(eye_right)
    nose = np.array(nose)

    package_coordinate['face'].append(face)
    package_coordinate['slip'].append(slip)
    package_coordinate['eye_left'].append(eye_left)
    package_coordinate['eye_right'].append(eye_right)
    package_coordinate['nose'].append(nose)

    mask = np.ones_like(img) * 255

    cv2.drawContours(mask, [face], -1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.drawContours(mask, [eye_right], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask, [eye_left], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask, [slip], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ellipse = cv2.fitEllipse(nose)
    cv2.ellipse(mask, ellipse, (255, 255, 255), -1, cv2.LINE_AA)

    # Áp dụng mask lên ảnh gốc để cắt vùng theo đa giác
    result_img = cv2.bitwise_or(img, mask)
    return result_img, package_coordinate

def CropDlib(img1, package_coordinate):
    img = np.array(img1)
    shape = predictor(img, dlib.rectangle(left=0, top=0, right=int(img1.size[0]), bottom=int(img1.size[1])))
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    
    forehead = cropForehead(img, landmarks)
    eyes = cropEye(img, landmarks)
    smile = cropSmile(img, landmarks)
    crop_face_remove_eye_nose, package_coordinate = crop_face_remove_eye_slip_nose(img, landmarks, package_coordinate)

    # Chuyển đổi từ mảng numpy sang đối tượng Image
    forehead_image = Image.fromarray(forehead)
    eyes_image = Image.fromarray(eyes)
    smile_image = Image.fromarray(smile)
    crop_face_remove_eye_nose_image = Image.fromarray(crop_face_remove_eye_nose)

    # Lưu ảnh
    # forehead_image.save("debug/forehead.png")
    # eyes_image.save("debug/eyes.png")
    # smile_image.save("debug/smile.png")
    # crop_face_remove_eye_nose_image.save("debug/crop_face_remove_eye_nose.png")

    return forehead, eyes, smile, crop_face_remove_eye_nose, package_coordinate
    