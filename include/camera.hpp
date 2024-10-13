#pragma once

#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera(int device = 0, int width = 640, int height = 480);
    ~Camera();

    bool isOpened() const;
    bool getFrame(const cv::Mat& frame);

private:
    cv::VideoCapture mCap;
};
