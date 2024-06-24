from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import numpy as np
from keras.models import load_model
import dlib

app = Flask(__name__)

# Directorio actual de la aplicación Flask
base_dir = os.path.abspath(os.path.dirname(__file__))

# Directorio de subida y configuración en app
UPLOAD_FOLDER = os.path.join(base_dir, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Rutas relativas a los modelos
model_path = os.path.join(base_dir, 'models', 'bts_model_final4.keras')
predictor_path = os.path.join(base_dir, 'models', 'shape_predictor_68_face_landmarks.dat')

# Cargar el modelo preentrenado
model = load_model(model_path)

# Nombre de los miembros de BTS
bts_members = ['Jin', 'Suga', 'J-Hope', 'RM', 'Jimin', 'V', 'Jungkook']

# Detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Predictor de landmarks de dlib
predictor = dlib.shape_predictor(predictor_path)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

def recognize_faces(image, faces):
    recognized_faces = []
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        predictions = model.predict(face_img)
        name_index = np.argmax(predictions[0])
        name = bts_members[name_index]

        recognized_faces.append((x, y, w, h, name))
    return recognized_faces

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            faces = detect_faces(image)
            recognized_faces = recognize_faces(image, faces)

            for (x, y, w, h, name) in recognized_faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_path, image)
            return send_from_directory(app.config['UPLOAD_FOLDER'], 'result_' + filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
