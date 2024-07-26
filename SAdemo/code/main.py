from flask import Flask, render_template, request
from PIL import Image
import base64
from io import BytesIO
from Classification import Classification
from DlibCrop import CropDlib
from yolov5_face.Yolov5face import GetFace
from yolov8_seg.predict import predict_yolov8_seg
from unet.Unet import predict_wrinkle
import numpy as np
    

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', alert_message='No file uploaded'), 400
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', alert_message='No file selected'), 400
        if file:
            # Read File
            input_image = Image.open(file.stream)
            
            # Get Face
            num_face, image_face = GetFace(input_image)
            if(num_face == 0):
                return render_template('index.html', alert_message='There are no faces in the image. Please choose a picture that has one face in it.'), 200
            elif (num_face > 1):
                return render_template('index.html', alert_message="There are {} faces in the image. Please choose a picture that has only one face in it.".format(num_face)), 200
            else:
                
                input_image = Image.fromarray(image_face)
                
                # preprocess data
                package_coordinate = {}
                Forehead, Eyes, Smile, Crop_face_remove_eye_nose, package_coordinate = CropDlib(input_image, package_coordinate)
                
                # Classification SA
                Pigment = Classification(input_image, "Pigmentation")
                Pores = Classification(input_image, "Pores")
                input_forehead = Image.fromarray(Forehead)
                foreheadW = Classification(input_forehead, "ForeheadWrinkle")
                input_eyes = Image.fromarray(Eyes)
                EyesW = Classification(input_eyes, "EyesWrinkle")
                input_smile = Image.fromarray(Smile)
                SmileW = Classification(input_smile, "SmileWrinkle")
                
                img_1 = BytesIO()
                input_image.save(img_1, 'PNG')
                img_1.seek(0)
                image1_base64 = base64.b64encode(img_1.getvalue()).decode('utf-8')
                
                # Segment pores
                output_seg_pores = Image.fromarray(predict_yolov8_seg(input_image, 'pores'))
                img_2 = BytesIO()
                output_seg_pores.save(img_2, 'PNG')
                img_2.seek(0)
                image2_base64 = base64.b64encode(img_2.getvalue()).decode('utf-8')
                
                input_wrinkle_seg = Image.fromarray(Crop_face_remove_eye_nose)
                output_seg_wrinkle = Image.fromarray(predict_wrinkle(input_image, input_wrinkle_seg, package_coordinate))
                img_3 = BytesIO()
                output_seg_wrinkle.save(img_3, 'PNG')
                img_3.seek(0)
                image3_base64 = base64.b64encode(img_3.getvalue()).decode('utf-8')
        
                return render_template('result.html', 
                                    image1=image1_base64,
                                    image2=image2_base64,
                                    image3=image3_base64,
                                    Pigment = Pigment, 
                                    Pores = Pores, 
                                    foreheadW = foreheadW,
                                    eyesW = EyesW,
                                    smileW = SmileW,
                                    ), 200
    return render_template('index.html')

if __name__ == '__main__':
     app.run(host='127.0.0.1', debug=True, port=5100)
