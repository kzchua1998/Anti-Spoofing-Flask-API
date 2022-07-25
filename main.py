import os
import cv2
import numpy as np
import argparse
import warnings
import time
import base64
from PIL import Image
from io import BytesIO

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

from flask import Flask, request
from flask_restful import Api, Resource


SAMPLE_IMAGE_PATH = "./images/sample/"


"""
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True"""


def model_prediction(image, model_dir, device_id, threshold):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    #image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    """result = check_image(image)
    if result is False:
        return"""
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    # draw result of prediction
    #print(prediction)
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1 and value > threshold:
        #print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        #result_text = "Real Score: {:.2f}".format(value)
        #color = (0, 255, 0)
        label = 1
    else:
        #print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        #result_text = "Fake Score: {:.2f}".format(value)
        #color = (0, 0, 255)
        label = 0
    #print("Prediction cost {:.2f} s".format(test_speed))
    #cv2.rectangle(image,(image_bbox[0], image_bbox[1]),(image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),color, 2)
    #cv2.putText(image,result_text,(image_bbox[0], image_bbox[1] - 5),cv2.FONT_HERSHEY_COMPLEX, 1.0*image.shape[0]/1024, color)
    #print(prediction)
    #format_ = os.path.splitext(image_name)[-1]
    #result_image_name = image_name.replace(format_, "_result" + format_)
    #cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    return label, value

model_dir = 'resources/anti_spoof_models'

app = Flask(__name__)
api = Api(app)

#@app.route('/process', methods = ['POST'])
class Process(Resource):
    def post(self):
        print(request.is_json)
        if request.is_json == True:
            content = request.json
            #print('here')
            img = np.array(Image.open(BytesIO(base64.b64decode(content['data']))))
            label, value = model_prediction(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),model_dir,0,0.0)
            print({'result': label, 'confidence': value})
            return {'result': label, 'confidence': value}

api.add_resource(Process, "/process")
app.run(debug=False,host='0.0.0.0')
