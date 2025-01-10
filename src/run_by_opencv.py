from datetime import datetime # Just for logging
import sys, traceback, logging # Just for logging
import pyrealsense2 as rs
import numpy as np
import cv2
import os


##### Logger start
# Get the current date in YYYY-MM-DD format for file organization
current_date = datetime.now().strftime("%Y-%m-%d")

# Define the folder path and create it if it doesn't exist
log_folder = os.path.join(os.getcwd(), "..", "logs", current_date)
os.makedirs(log_folder, exist_ok=True)

# Define the log file path
log_file = os.path.join(log_folder, "app.log")

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,  # Log everything from DEBUG and above
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Write logs to a file
        logging.StreamHandler()        # Also output logs to the terminal
    ]
)

# Example usage
##### Logger end

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Align depth to color
align = rs.align(rs.stream.color)

# Start the camera
pipeline.start(config)

# Load YOLO model using OpenCV's DNN module
model_cfg = "cfg/deploy/yolov7.yaml"  # Replace with your YOLO model config file
model_weights = "weights/best.pt"  # Replace with your trained YOLO model weights file
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

# Set the backend and target for DNN module (CPU or GPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get layer names and set output layer
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

try:
    while True:
        # Wait for a coherent pair of frames: color and depth
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Perform object detection on the color image using OpenCV DNN
        blob = cv2.dnn.blobFromImage(color_image, 1/255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Process detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Threshold for detection confidence
                    center_x = int(obj[0] * color_image.shape[1])
                    center_y = int(obj[1] * color_image.shape[0])
                    width = int(obj[2] * color_image.shape[1])
                    height = int(obj[3] * color_image.shape[0])

                    # Get bounding box coordinates
                    x1 = int(center_x - width/2)
                    y1 = int(center_y - height/2)
                    x2 = int(center_x + width/2)
                    y2 = int(center_y + height/2)

                    # Extract depth region
                    depth_region = depth_image[y1:y2, x1:x2]
                    depth_values = depth_region.flatten()

                    # Calculate the median depth
                    if len(depth_values) > 0:
                        object_depth = np.median(depth_values)
                    else:
                        object_depth = 0  # Fallback in case of invalid depth data

                    # Label with class and depth information
                    label = f"Class {class_id} ({object_depth} mm)"

                    # Draw bounding box and depth label
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with bounding boxes and depth information
        cv2.imshow("Color Image with Depth Annotations", color_image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
