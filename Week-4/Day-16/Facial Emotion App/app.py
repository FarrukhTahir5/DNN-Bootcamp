from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff','jfif'}

# Load the pre-trained model
json_file = open("model.json", "r")
loaded_json_model = json_file.read()
json_file.close()
model = model_from_json(loaded_json_model)
model.load_weights("model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img = roi_gray.reshape((1, 48, 48, 1))
                img = img / 255.0
                max_index = np.argmax(model.predict(img.reshape((1, 48, 48, 1))), axis=-1)[0]
                predicted_emotion = emotions[max_index]
                cv2.putText(image, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "output_" + filename)
            cv2.imwrite(output_filepath, image)

            return render_template("index.html", image_url=output_filepath, emotion=predicted_emotion)
    return render_template("index.html")

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
