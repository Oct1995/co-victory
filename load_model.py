
import numpy as np
import pandas as pd
from flask import Flask, render_template, request ,jsonify,url_for,redirect, flash
import tensorflow as tf
import cv2
import os

# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/data'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
asin_title_dict = None
model = None
# train = pd.read_csv('static/data/train_df.csv')
# cat = pd.read_csv('static/data/categories.csv')
# categories = cat[['scale_id','category_description','category']]
# df_select = train[['title','job_id','scale_id','scale_name','element_name','data_value','category']]

# job_code = train['job_id'].unique().tolist()
# title = train['title'].unique().tolist()

if asin_title_dict is None or model is None:
    model = tf.keras.models.load_model('xray_model.h5')
    
# # L2-normalized vectors.

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare(filepath,model):

    img = cv2.imread(filepath)

    img = cv2.resize(img,(150,150))

    img = np.reshape(img,[1,150,150,3])

    classes = model.predict_classes(img.astype(float) / 255.0)

    
    return classes[0][0]

# CATEGORIES = [ 'covid','normal'] 
# print(CATEGORIES[prepare('1.jpg',model_1)])


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route('/page', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            a = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            CATEGORIES = [ 'COVID-19','Normal'] 
            result = CATEGORIES[prepare(a,model)]
            phase = request.form.get("1")
            a1 = request.form.get("2")
            a2 = request.form.get("3")
            a3 = request.form.get("4")
            a4 = request.form.get("5")
            a5 = request.form.get("6")
            a = []
            if a1 != None:
                a.append(int(a1))
            if a2 != None:
                a.append(int(a2))
            if a3 != None:
                a.append(int(a3))
            if a4 != None:
                a.append(int(a4))
            if a5 != None:
                a.append(int(a5))
            if phase != None:
                a.append(int(phase))
            if sum(a) >= 4:
                Q = ''' \n ask the patient to perform hand hygiene,
                \n wear a surgical mask,
                \n direct the patient through the respiratory pathway 
                and inform MD for assessment.'''
            else:
                Q = 'the patient had not been daignosed by COVID-19 by the Checklist score'
            
            return render_template('model.html', file = result , phase  =Q.splitlines() ,  a =sum(a) )
                                  
    return render_template('model.html')




if __name__ == '__main__':
    app.run(debug=True)

