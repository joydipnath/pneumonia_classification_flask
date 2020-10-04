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
        # densenet_bin_sigmoid_model = load_model(app.config['MODEL_PATH'] + "densenet_model_binary_sigmoid_focalloss.h5",compile=False)

        Final_DenseNet121_model = load_model(app.config['MODEL_PATH'] + "Final_DenseNet121_model.h5", compile=False)
        Final_DenseNet121_model.load_weights(app.config['MODEL_PATH'] + "Final_DenseNet121_weights.h5")
        Final_InceptionV3_model = load_model(app.config['MODEL_PATH'] + "Final_InceptionV3_model.h5", compile=False)
        Final_InceptionV3_model.load_weights(app.config['MODEL_PATH'] + "Final_InceptionV3_weights.h5")

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

        # Model 1 - DenseNet121
        pred1 = Final_DenseNet121_model.predict(image_data)
        # Model 2 - InceptionV3
        pred2 = Final_InceptionV3_model.predict(image_data)

        # Select Best Model prediction
        model1_pred = "Pneumonic" if np.argmax(pred1) == 1 else "Normal"
        model2_pred = "Pneumonic" if np.argmax(pred2) == 1 else "Normal"

        showMasks = False
        print(pred1)
        print(pred2)

        if np.argmax(pred1) > np.argmax(pred2):  # If Pred1 is Pneumonic, Pred2 is Normal
            prediction = model1_pred
            pred = pred1[0]
            showMasks = True
        elif np.argmax(pred1) == np.argmax(pred2 == 1):  # If Pred1 and Pred 2 are Pneumonic
            showMasks = True

            if pred1[0][1] > pred2[0][1]:
                prediction = model1_pred
                pred = pred1[0]
            else:
                prediction = model2_pred
                pred = pred2[0]
        elif np.argmax(pred1) == np.argmax(pred2 == 0):  # If Pred1 and Pred 2 are Normal
            showMasks = False
            if pred1[0][0] > pred2[0][0]:
                prediction = model1_pred
                pred = pred1[0]
            else:
                prediction = model2_pred
                pred = pred2[0]
        else:  # If Pred2 is Normal and Pred2 is Pneumonic
            showMasks = True
            prediction = model2_pred
            pred = pred2[0]

        # print("Model Prediction:", prediction)
        # print("     Normal    : %.2f%% \n     Pneumonic: %.2f%%" % (100 * pred[0], 100 * pred[1]))

        return prediction, image_name, [100 * pred[0], 100 * pred[1]]


    def load_model_to_app(self):
        model = load_model('densenet_model_binary_sigmoid_focalloss.h5')
        return model
