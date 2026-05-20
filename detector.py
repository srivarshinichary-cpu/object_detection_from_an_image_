import os
from pathlib import Path
import cv2
import numpy as np

try:
	import tensorflow as tf
except Exception:
	tf = None

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'ssd_mobilenet_v2_fpnlite_320x320' / 'saved_model'
model = None
if tf is not None and MODEL_DIR.exists():
	model = tf.saved_model.load(str(MODEL_DIR))

labels_path = BASE_DIR / 'labels.txt'
class_names = []
if labels_path.exists():
	with open(labels_path, 'r', encoding='utf-8') as f:
		class_names = [l.strip() for l in f.readlines() if l.strip()]

def detect_objects(image_path, score_threshold=0.5):
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Image not found: {image_path}")
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	detected_labels = []
	output_path = None
	if model is not None:
		input_tensor = tf.convert_to_tensor(image_rgb)
		input_tensor = input_tensor[tf.newaxis, ...]
		detections = model(input_tensor)
		boxes = detections.get('detection_boxes')[0].numpy()
		classes = detections.get('detection_classes')[0].numpy().astype(np.int32)
		scores = detections.get('detection_scores')[0].numpy()
		h, w = image.shape[:2]
		for i in range(len(scores)):
			if scores[i] >= score_threshold:
				class_id = int(classes[i])
				label = class_names[class_id - 1] if 0 < class_id <= len(class_names) else f'Class {class_id}'
				detected_labels.append(label)
				y1, x1, y2, x2 = boxes[i]
				x1 = int(x1 * w); y1 = int(y1 * h); x2 = int(x2 * w); y2 = int(y2 * h)
				cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
				cv2.putText(image, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		output_path = str(Path(image_path).with_name(Path(image_path).stem + "_result" + Path(image_path).suffix))
		cv2.imwrite(output_path, image)
	else:
		# No model available; save a copy and return no detections
		output_path = str(Path(image_path).with_name(Path(image_path).stem + "_result" + Path(image_path).suffix))
		cv2.imwrite(output_path, image)
	return output_path, detected_labels