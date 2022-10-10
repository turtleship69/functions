import logging

import azure.functions as func

import cv2
import numpy as np
import base64
from io import BytesIO

def sketcher(input):
    img = get_opencv_img_from_buffer(input, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray_image = 255 - gray_image
    blurred_img = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    inverted_blurred_img = 255 - blurred_img
    pencil_sketch_IMG = cv2.divide(gray_image,
                                   inverted_blurred_img,
                                   scale=256.0)
    #source: https://python.plainenglish.io/convert-a-photo-to-pencil-sketch-using-python-in-12-lines-of-code-4346426256d4e
    return pencil_sketch_IMG

def get_opencv_img_from_buffer(buffer, flags):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger 2 function processed a request.')

    #convert image to sketch from request.files['image']
    sketched_image = sketcher(req.files['image'].stream)

    #convert sketch to image that can be returned to browser
    sketched_image = cv2.imencode('.png', sketched_image)[1].tostring()

    #return base64 encoded image with mimetypes
    return func.HttpResponse(BytesIO(sketched_image).read(), mimetype="image/png")