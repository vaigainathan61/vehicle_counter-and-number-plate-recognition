import sys
import os

# Add the sort directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))

from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort  # Corrected import statement
from util import get_car, read_license_plate, write_csv

# Initialize results dictionary and SORT tracker
results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Load video
cap = cv2.VideoCapture('sample.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define vehicle classes (coco dataset indices for car, motorcycle, bus, and truck)
vehicles = [2, 3, 5, 7]

# Process each frame
frame_nmr = -1
while True:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is read (end of video)

    results[frame_nmr] = {}
    # Detect vehicles
