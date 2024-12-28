from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Replace with your model path if different

# Define a function for object detection
def detect_objects(frame):
    results = model(frame)  # Perform detection
    detections = []

    # Iterate over the detections
    for result in results:  # Each result corresponds to a single frame
        for box in result.boxes:  # Iterate over detected boxes
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = f"{model.names[cls]} {confidence:.2f}"  # Label with class and confidence
            
            # Draw rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store detection information
            detections.append({
                'label': label,
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence
            })

    return frame, detections

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Could not read image'}), 400

    # Perform object detection
    frame, detections = detect_objects(frame)

    # Encode the frame to send back as a response
    _, buffer = cv2.imencode('.jpg', frame)
    response_frame = buffer.tobytes()

    # Convert the image to a base64 string for JSON response
    response_frame_base64 = base64.b64encode(response_frame).decode('utf-8')

    return jsonify({'detections': detections, 'image': response_frame_base64})

if __name__ == '__main__':
    app.run(debug=True)