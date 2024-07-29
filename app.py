from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from lime import lime_image
from skimage.segmentation import slic
import torch
import base64

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model
model_path = "models/640.pt"  # Update this to the actual path of your model file
model = YOLO(model_path)

# Define class names
class_names = [
    'Blue-Tailed-Day-Gecko', 'Echo-Parakeet', 'Mascarene-Paradise-Flycatcher', 'Mauritius-Black-Bulbul',
    'Mauritius-Cuckooshrike', 'Mauritius-Flying-Fox', 'Mauritius-Fody', 'Mauritius-Grey-White-Eye',
    'Mauritius-Kestrel', 'Mauritius-Lowland-Forest-Day-Gecko', 'Mauritius-Olive-White-Eye',
    'Mauritius-Snake-eyed-Skink', 'Ornate-Day-Gecko', 'Pink-Pigeon', 'Round-Island-Day-Gecko',
    'Round-Island-Keel-Scaled-Boa', 'Round-Island-Skink-Telfair-s-Skink'
]

@app.route('/')
def index():
    return "Welcome to the YOLOv8 and LIME Explanation API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        results = model(img)

        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    'box': box.xyxy.tolist()[0],  # Bounding box coordinates
                    'class': class_names[int(box.cls.tolist()[0])],  # Map index to label
                    'confidence': float(box.conf.tolist()[0])  # Confidence score
                }
                detections.append(detection)

        response = {
            'detections': detections
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_array = np.array(img)

        # Custom segmentation function for LIME
        def custom_segmentation_fn(image):
            return slic(image, n_segments=100, compactness=10, sigma=1)

        # Define the prediction function for LIME
        def yolo_predict(images):
            results = []
            for img in images:
                img = Image.fromarray(img.astype('uint8'), 'RGB')
                img = np.array(img.resize((640, 640)))
                img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                preds = model(img_tensor)
                probs = np.zeros(len(class_names))  # Number of classes
                if preds[0] is not None and len(preds[0].boxes) > 0:
                    for box in preds[0].boxes:
                        class_idx = int(box.cls)
                        conf = box.conf.item()
                        probs[class_idx] = max(probs[class_idx], conf)
                results.append(probs)
            return np.array(results)

        # Define the LIME image explainer
        explainer = lime_image.LimeImageExplainer()

        # Generate explanation
        explanation = explainer.explain_instance(img_array, yolo_predict, top_labels=1, hide_color=0, num_samples=100, segmentation_fn=custom_segmentation_fn)

        # Get the explanation image for the class with the highest confidence score
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)
        img_boundry = mark_boundaries(temp / 255.0, mask, color=(1, 0, 0), outline_color=(0, 1, 0))

        # Convert explanation image to base64
        img_pil = Image.fromarray((img_boundry * 255).astype(np.uint8))
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = {
            'explanation': img_str,
            'class': class_names[top_label],
            'message': (
                f"The LIME explanation highlights the regions that most strongly contribute to the model's prediction of the '{class_names[top_label]}'. "
                "These highlighted regions are areas that the model found most relevant for identifying this class. "
                "The explanation provides insights into which parts of the image were used by the model to make the prediction. "
                "However, it's important to note that the model might sometimes focus on regions that may not seem intuitive to humans."
            )
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
