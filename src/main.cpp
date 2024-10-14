#include "InputCamera.hpp"
#include "ModelONNX.hpp"
#include "Inference.hpp"

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    std::unique_ptr<Input> input = std::make_unique<InputCamera>(0);
    std::unique_ptr<Model> model = std::make_unique<ModelONNX>("models/yolo11n.onnx");

    if (!input->inputWorking()) {
        std::cout << "Error: Unable to open the camera." << std::endl;
        return -1;
    }

    //if (!model->isLoaded()) {
    //    std::cout << "Error: Unable to load the model." << std::endl;
    //    return -1;
    //}
    Inference inf("C:\\projects\\real-time_object_detection\\models\\yolov8n.onnx", cv::Size(640, 640), "classes.txt", false);

    cv::Mat frame;
    while (true) { 
        if (!input->getFrame(frame)) {
            std::cout << "Error: Unable to capture frame." << std::endl;
            break;
        }

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }

        // model->applyModel(frame);  

        cv::imshow("Real-Time Object Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
