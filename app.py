from flask import Flask, render_template, request
import numpy as np
import os
import threading
import time
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load TFLite model
MODEL_PATH = 'model.tflite'
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Sehat', 'Antraknosa', 'Virus Kuning', 'Bercak Daun', 'Keriting Daun']

os.makedirs('static/uploads', exist_ok=True)

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype("float32")

    img = img / 127.5 - 1.0

    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_path):
    img_array = preprocess(img_path)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])

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

if __name__ == '__main__':
    app.run()
