# Real Time Object Detection 
This project builds a real time object detector. 

The project is based on two main parts, an Input class and a Model class. The model class is implemented using dependency injection to enable easy expansion and development of alternative models. 

# Author
Johannes Källstad [EMail](johannes.kallstad@gmail.com) [GitHub](https://github.com/ohukainen)

# Dependencies
- [Conan](https://conan.io/)
- [CMake](https://cmake.org/)
- [OpenCV](https://opencv.org/)
- [Ultralytics](https://www.ultralytics.com/) 

# Build
If unfamiliar with conan basic usage, see [documantation](https://docs.conan.io/2/tutorial/consuming_packages/build_simple_cmake_project.html).

# Scripts 
To create the yolov8n.onnx file run the [YOLO_to_ONNX script](scripts/YOLO_to_ONNX.py). 

# Notice 
Parts of this project is based on examples from the [Ultralytics GitHub page](https://github.com/ultralytics/ultralytics), notices can be found in relevant files.  
