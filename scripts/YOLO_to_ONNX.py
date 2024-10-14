# Copied from https://docs.ultralytics.com/integrations/onnx/#installation
from ultralytics import YOLO

model_name = "yolov8n.pt"

# Load the YOLO11 model
model = YOLO(model_name)

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'
