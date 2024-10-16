#include "InputCamera.hpp"
#include "ModelYOLO.hpp"

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    std::unique_ptr<Input> input = std::make_unique<InputCamera>(0);
    std::unique_ptr<Model> model = std::make_unique<ModelYOLO>("C:\\projects\\real-time_object_detection\\models\\yolov8n.onnx", cv::Size(640, 640), false);

    if (!input->inputWorking()) {
        std::cout << "Error: Unable to open the camera." << std::endl;
        return -1;
    }

    if (!model->isLoaded()) {
        std::cout << "Error: Unable to load the model." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) { 
        if (!input->getFrame(frame)) {
            std::cout << "Error: Unable to capture frame." << std::endl;
            break;
        }

        cv::flipND(frame, frame, 1);

        std::vector<Detection> output = model->applyModel(frame);

        model->drawDetections(frame, output);

        cv::imshow("Real-Time Object Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
