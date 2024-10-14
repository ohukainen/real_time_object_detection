#pragma once

#include "Input.hpp"

class InputCamera : public Input {
public:
    InputCamera(int device);
    ~InputCamera() = default;

    InputCamera(const InputCamera &) = delete;
    InputCamera(InputCamera &&) = default;
    InputCamera &operator=(const InputCamera &) = delete;
    InputCamera &operator=(InputCamera &&) = default;

    bool inputWorking() override;
    bool getFrame(cv::Mat& frame) override;

private:
    cv::VideoCapture mCap;
};
