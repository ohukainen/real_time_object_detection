#include "InputCamera.hpp"

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    std::unique_ptr<Input> input = std::make_unique<InputCamera>(0);

    if (!input->inputWorking()) {
        std::cout << "Error: Unable to open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        if (!input->getFrame(frame)) {
            std::cout << "Error: Unable to capture frame." << std::endl;
            break;
        }

        cv::imshow("Real-Time Object Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
