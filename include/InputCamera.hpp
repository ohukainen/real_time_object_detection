#pragma once

#include "Input.hpp"

#include <opencv2/opencv.hpp>

class InputCamera : public Input {
public:
    InputCamera(int device);
    ~InputCamera() = default;

    bool inputWorking() override;
    bool getFrame(cv::Mat& frame) override;

private:
    cv::VideoCapture mCap;
};
