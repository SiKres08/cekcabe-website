from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import os
import threading
import time

app = Flask(__name__)

MODEL_PATH = 'model_cnn.keras'
model = load_model(MODEL_PATH)

class_names = ['Sehat', 'Antraknosa', 'Virus Kuning', 'Bercak Daun', 'Keriting Daun']

# Pastikan folder ada
os.makedirs('static/uploads', exist_ok=True)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    accuracy = np.max(preds) * 100

    return predicted_class, accuracy

def delete_later(path):
    time.sleep(60)
    if os.path.exists(path):
        os.remove(path)

@app.route("/")
def landing():
    return render_template("index.html")

@app.route("/cekcabe")
def cekcabe():
    return render_template("cekcabe.html")

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    import uuid
    filename = str(uuid.uuid4()) + "_" + file.filename

    file_path = os.path.join('static/uploads', filename)
    file.save(file_path)

    threading.Thread(target=delete_later, args=(file_path,)).start()

    predicted_class, accuracy = predict_image(file_path)

    image_url = f"uploads/{filename}"

    return render_template('result.html',
                           prediction=predicted_class,
                           accuracy=round(accuracy, 2),
                           image_path=image_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
