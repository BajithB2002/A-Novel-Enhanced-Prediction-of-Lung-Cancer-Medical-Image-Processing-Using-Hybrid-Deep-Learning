
from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from keras.models import load_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

model1 = load_model(
    'models/ct_cnn_best_model.hdf5')
model2 = load_model(
    'models/ct_incep_best_model.hdf5')
model3 = load_model(
    'models/ct_resnet_best_model.hdf5')
model4 = load_model(
    'models/ct_vgg_best_model.hdf5')
import warnings
warnings.filterwarnings("ignore")
UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/news.html')
def news():
   return render_template('news.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/faqs.html')
def faqs():
   return render_template('faqs.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')



@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   models = [model1, model2, model3, model4]
   classes_dir = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
   path = './flask app/assets/images/upload_chest.jpg'
   img = image.load_img(path, target_size=(350, 350))
   # Normalizing Image
   norm_img = image.img_to_array(img) / 255
   # Converting Image to Numpy Array
   input_arr_img = np.array([norm_img])
   # Getting Predictions
   preds = [model.predict(input_arr_img) for model in models]
   preds = np.array(preds)
   summed = np.sum(preds, axis=0)
   ensemble_prediction = np.argmax(summed, axis=1)
   lst = summed.tolist()
   lst = lst[0]
   print(classes_dir[(lst.index(max(lst)))])


   return render_template('results_chest.html',inception_chest_pred=classes_dir[(lst.index(max(lst)))])



if __name__ == '__main__':
   app.secret_key = ".."
   app.run(host='127.0.0.1', port=8005, debug=True)