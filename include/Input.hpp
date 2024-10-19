#pragma once

#include <opencv2/opencv.hpp>

class Input {
public:
    Input(int device);
    Input(const std::string& device);

    ~Input() = default;

    bool capturing();
    bool getFrame(cv::Mat& frame);

    bool isVideo() const {return mIsVideo;};

private:
    cv::VideoCapture mCap;
    bool mIsVideo;
};
