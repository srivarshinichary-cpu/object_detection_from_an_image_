import os
import base64
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

MODEL_DIR = 'ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'
detect_fn = None
try:
    import tensorflow as tf
    detect_fn = tf.saved_model.load(MODEL_DIR)
except Exception as e:
    print(f"Warning: failed to load model '{MODEL_DIR}': {e}")
    tf = None

category_index = {}
if os.path.exists('labels.txt'):
    with open('labels.txt', 'r', encoding='utf-8') as f:
        category_index = {i + 1: name.strip() for i, name in enumerate(f) if name.strip()}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('index.html', result='No image uploaded.')

    if not allowed_file(file.filename):
        return render_template('index.html', result='Invalid file format.')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = Image.open(filepath)
    image_np = np.array(image.convert('RGB'))

    if detect_fn is None or tf is None:
        return render_template('index.html', result='Detection model is not available.')

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    height, width, _ = image_np.shape
    detected_classes = []

    for i in range(len(scores)):
        if scores[i] > 0.5:
            class_id = int(classes[i])
            class_name = category_index.get(class_id, 'unknown')
            detected_classes.append(class_name)

            y_min, x_min, y_max, x_max = boxes[i]
            left = int(x_min * width)
            right = int(x_max * width)
            top = int(y_min * height)
            bottom = int(y_max * height)

            cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image_np, class_name, (left, max(top - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    query = request.form.get('query', '').strip().lower()
    if query:
        found = any(query == cls.lower() for cls in detected_classes)
        result_text = f'Found: {query}' if found else f'Not found: {query}'
    else:
        result_text = 'No query provided.'

    _, buffer = cv2.imencode('.jpg', image_np)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html', result=result_text, image_data=encoded_img)


if __name__ == '__main__':
    app.run(debug=True)
 