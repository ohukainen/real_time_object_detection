#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {

    cv::VideoCapture cap;

    cap.open(0);
    if (!cap.isOpened()) {
        std::cout << "Error: Unable to open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
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
