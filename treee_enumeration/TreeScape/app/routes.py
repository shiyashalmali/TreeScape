from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from ultralytics import YOLO
import cv2
import os
from PIL import Image
from io import BytesIO
import base64
import random

main = Blueprint('main', __name__)

# Configure folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load models
DETECTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "PythonScripts/count.pt")
CLASSIFICATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "PythonScripts/species.pt")

detection_model = YOLO(DETECTION_MODEL_PATH)
classification_model = YOLO(CLASSIFICATION_MODEL_PATH)

# Basic routes
@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict')
def predict():
    return render_template('predict.html')

@main.route('/tree_count')
def tree_count():
    return render_template('tree_count.html')

@main.route('/object_segmentation')
def object_segmentation():
    return render_template('object_segmentation.html')

@main.route('/type_count')
def type_count():
    return render_template('type_count.html')

@main.route('/map_boundary')
def map_boundary():
    return render_template('map_boundary.html')

@main.route('/historical_data')
def historical_data():
    return render_template('historical_data.html')

@main.route('/optimal_path')
def optimal_path():
    return render_template('optimal_path.html')

# Tree counting endpoint
@main.route('/count-trees', methods=['POST'])
def count_trees():
    print('Processing tree count request...')
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Ensure directories exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)

        # Decode and save the uploaded image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        upload_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
        image.save(upload_path)
        print(f"Saved uploaded image to: {upload_path}")

        # Perform YOLO inference
        results = detection_model(upload_path)
        
        if len(results) > 0:
            tree_count = len(results[0].boxes)
            
            # Save the processed image
            save_path = os.path.join(RESULTS_FOLDER, 'processed_image.png')
            result_image = results[0].plot()  # Get the plotted image
            
            # Convert numpy array to PIL Image and save
            if result_image is not None:
                result_pil = Image.fromarray(result_image)
                result_pil.save(save_path)
                print(f"Saved processed image to: {save_path}")
            else:
                # Fallback: save the original image with bounding boxes
                results[0].save(save_path)
        else:
            tree_count = 0
            # If no trees detected, save the original image
            image.save(os.path.join(RESULTS_FOLDER, 'processed_image.png'))

        print(f"Detected {tree_count} trees")
        
        return jsonify({
            'tree_count': tree_count,
            'processed_image_url': 'http://127.0.0.1:5000/processed-image/processed_image.png'
        })

    except Exception as e:
        print(f"Error in count_trees: {str(e)}")
        return jsonify({'error': f"Backend Error: {str(e)}"}), 500

# Tree classification endpoint
@main.route('/classify-trees', methods=['POST'])
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
                cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_cv2, class_label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Update counts
                if class_label == "Coniferous Tree":
                    coniferous_count += 1
                elif class_label == "Deciduous Tree":
                    deciduous_count += 1

        # Save the processed image
        save_path = os.path.join(RESULTS_FOLDER, 'processed_image.jpg')
        cv2.imwrite(save_path, image_cv2)

        return jsonify({
            'coniferous_count': coniferous_count,
            'deciduous_count': deciduous_count,
            'processed_image_url': 'http://127.0.0.1:5000/processed-image/processed_image.jpg'
        })

    except Exception as e:
        return jsonify({'error': f"Backend Error: {str(e)}"}), 500

@main.route('/process_map_boundary', methods=['POST'])
def process_map_boundary():
    try:
        if request.is_json:
            data = request.get_json()
            boundary_data = data.get('boundaryData')
        else:
            boundary_data = request.form.get('boundaryData')

        if not boundary_data:
            return jsonify({
                "status": "error",
                "message": "No boundary data received"
            }), 400

        # Parse the boundary data if it's a string
        if isinstance(boundary_data, str):
            try:
                boundary_data = json.loads(boundary_data)
            except json.JSONDecodeError:
                return jsonify({
                    "status": "error",
                    "message": "Invalid boundary data format"
                }), 400

        print("Boundary Data Received:", boundary_data)
        
        return jsonify({
            "status": "success",
            "boundary": boundary_data
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Serve processed images
@main.route('/processed-image/<filename>')
def processed_image(filename):
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500