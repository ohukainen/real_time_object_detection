#pragma once

#include <opencv2/opencv.hpp>

class Input {
public:
    virtual ~Input() = default;

    virtual bool inputWorking() = 0;
    virtual bool getFrame(cv::Mat& frame) = 0;
};
