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
- [JSON for Modern C++](https://json.nlohmann.me/)

# Build
If unfamiliar with conan basic usage, see [documantation](https://docs.conan.io/2/tutorial/consuming_packages/build_simple_cmake_project.html).

# Scripts 
To create the yolov8n.onnx file run the [YOLO_to_ONNX script](scripts/YOLO_to_ONNX.py). 

# Classes json format 
The optional json file that defines the classes used should have the following format: 
```json
{
    "classes" : [ 
        "Class_1",
        "Class_2", 
        "Class_3",
        ...
        "Class_N"
    ]
}
```

# Notice 
Parts of this project is based on examples from the [Ultralytics GitHub page](https://github.com/ultralytics/ultralytics), notices can be found in relevant files.  
