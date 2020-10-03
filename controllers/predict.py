from flask import request, jsonify
# import tensorflow as tf
import numpy as np
from keras import backend
from keras.models import load_model
from keras.preprocessing import sequence, image
from keras.applications.inception_v3 import preprocess_input
import os
import json
import cv2
import glob
import pydicom
from app import app


class Predict:

    # def __init__(self):
    #     pass


    @staticmethod
    def pneumonia_classification(filename):

        IMAGE_WIDTH = 224
        IMAGE_HEIGHT = 224

        # model = load_model(project_path + "densenet_model_binary_sigmoid_focalloss.h5", custom_objects={'focal_loss_fixed': focal_loss()})
        # if you wish to just perform inference with your model and not further optimization or training your model,
        densenet_bin_sigmoid_model = load_model(app.config['MODEL_PATH'] + "densenet_model_binary_sigmoid_focalloss.h5",
                                                compile=False)
        # densenet_bin_softmax_model = load_model(app.config['MODEL_PATH'] + "densenet_model_binary.h5", compile=False)
        # model_inception = load_model(app.config['MODEL_PATH'] + 'model_inception.h5')

        dcm_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dcm_data = pydicom.read_file(dcm_file)
        # print(dcm_data.pixel_array)
        im = dcm_data.pixel_array

        # pylab.imshow(im, cmap=pylab.cm.gist_gray)
        # pylab.axis('off')
        # basename = os.path.basename(filename)
        image_name = os.path.splitext(filename)[0] + '.jpg'
        image_name = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        cv2.imwrite(image_name, im)
        img = image.load_img(image_name, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        image_data = preprocess_input(x)

        prediction = []

        prediction.append({'densenet_bin_sigmoid_model': densenet_bin_sigmoid_model.predict(image_data)[0]})
        # prediction.append({'densenet_bin_softmax_model': densenet_bin_softmax_model.predict(image_data)[0]})

        result = []
        for index in prediction:
            j = 0
            for i in index:
                # if index[i][j] == 1:
                    result.append({'Lung Opacity': index[i][j]})
                # else:
                    result.append({'Normal': index[i][j+1]})

        return result, image_name

        # prediction = model.predict(review)
        # print("Prediction (0 = negative, 1 = positive) = ", end="")
        # print("%0.4f" % prediction[0][0])
        # return prediction[0][0]


    def load_model_to_app(self):
        model = load_model('densenet_model_binary_sigmoid_focalloss.h5')
        return model