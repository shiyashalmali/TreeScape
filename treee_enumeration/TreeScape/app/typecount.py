from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
from PIL import Image
from io import BytesIO
import base64
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Paths to models
DETECTION_MODEL_PATH = "PythonScripts/count.pt"
CLASSIFICATION_MODEL_PATH = "PythonScripts/species.pt"

# Load models
detection_model = YOLO(DETECTION_MODEL_PATH)
classification_model = YOLO(CLASSIFICATION_MODEL_PATH)

@app.route('/classify-trees', methods=['POST'])
def classify_trees():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode and save the uploaded image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        upload_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        image.save(upload_path)

        # Perform tree detection
        results = detection_model.predict(source=upload_path, save=False, device='cpu', verbose=False)

        # Load the input image
        image_cv2 = cv2.imread(upload_path)

        # Initialize counts
        coniferous_count = 0
        deciduous_count = 0

        # Process each detected bounding box
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Crop detected region
                cropped_image = image_cv2[y1:y2, x1:x2]

                # Save cropped image temporarily
                temp_crop_path = os.path.join(UPLOAD_FOLDER, "temp_crop.jpg")
                cv2.imwrite(temp_crop_path, cropped_image)

                # Perform classification
                classification_result = classification_model.predict(
                    source=temp_crop_path, save=False, device='cpu', verbose=False
                )

                # Determine class label
                if classification_result and classification_result[0].probs is not None:
                    probs = classification_result[0].probs
                    max_index = probs.argmax()
                    class_label = classification_result[0].names[max_index]
                else:
                    class_label = random.choice(["Coniferous Tree", "Deciduous Tree"])

                # Annotate bounding box on the image
                cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(image_cv2, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Update counts
                if class_label == "Coniferous Tree":
                    coniferous_count += 1
                elif class_label == "Deciduous Tree":
                    deciduous_count += 1

        # Save the processed image
        save_path = os.path.join(RESULTS_FOLDER, 'processed_image.jpg')
        cv2.imwrite(save_path, image_cv2)

        # Return response
        return jsonify({
            'coniferous_count': coniferous_count,
            'deciduous_count': deciduous_count,
            "processed_image_url": "http://127.0.0.1:5000/processed-image/processed_image.jpg"
        })

    except Exception as e:
        return jsonify({'error': f"Backend Error: {str(e)}"}), 500

@app.route('/processed-image/<filename>')
def processed_image(filename):
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
