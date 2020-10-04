from flask import jsonify, request, render_template, flash, redirect, url_for, send_from_directory
# from flask_cors import CORS, cross_origin
from controllers.predict import Predict
from werkzeug.utils import secure_filename
import os
from app import app
import base64


@app.route('/', methods=["GET"])
def index():
    name = 'Pneumonia Detection'
    return render_template('index.html', name=name)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'dcm'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=["POST"])
def upload_image():
    name = 'Pneumonia Detection'
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        predict = Predict()
        prediction, image_name, percentage = predict.pneumonia_classification(filename)
        flash('Image successfully uploaded and displayed')
        with open(image_name, "rb") as image_file:
            image_name = base64.b64encode(image_file.read())
        return render_template('index.html', name=name, filename=image_name.decode('utf-8'), prediction=prediction, percentage=percentage)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return send_from_directory(app.static_folder, filename)
    # return redirect(url_for('storage', filename='uploads/' + filename), code=301)
