import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Align depth to color
align = rs.align(rs.stream.color)

# Start the camera
pipeline.start(config)

# Load YOLO model
model = YOLO("yolov5s.pt")  # Replace with your trained YOLO model

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

        # Perform object detection on the color image
        results = model.predict(color_image)

        # Annotate the color image with depth information
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                depth_region = depth_image[y1:y2, x1:x2]
                depth_values = depth_region.flatten()

                # Calculate the median depth
                if len(depth_values) > 0:
                    object_depth = np.median(depth_values)
                else:
                    object_depth = 0  # Fallback in case of invalid depth data

                label = f"{box.cls} ({object_depth} mm)"

                # Draw bounding box and depth information
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    color_image, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )

        # Display the image
        cv2.imshow("Color Image with Depth Annotations", color_image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
