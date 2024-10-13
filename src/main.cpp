#include "camera.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {

    Camera camera(0, 640, 480);
    if (!camera.isOpened()) {
        std::cout << "Error: Unable to open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        if (!camera.getFrame(frame)) {
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
