from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for testing

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)  # Create results folder if it doesn't exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_WEIGHTS_PATH="PythonScripts/count.pt"

model = YOLO(MODEL_WEIGHTS_PATH)

@app.route('/count-trees', methods=['POST'])
def count_trees():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode and save the uploaded image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        upload_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
        image.save(upload_path)

        # Perform YOLO inference
        results = model(upload_path)
        if len(results) > 0:
            tree_count = len(results[0].boxes)  # Count detected boxes
        else:
            tree_count = 0

        # Save the processed image
        save_path = os.path.join(RESULTS_FOLDER, 'processed_image.png')
        results[0].plot(save_path)

        # Return response
        return jsonify({
            'tree_count': tree_count,
            # 'processed_image_url': f'http://127.0.0.1:5000/processed-image/{os.path.basename(save_path)}'
            "processed_image_url": "http://127.0.0.1:5000/processed-image/processed_image.png"
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


