#pragma once 

#include <opencv2/opencv.hpp>

class Model {
public:
    virtual ~Model() = default;

    virtual bool isLoaded() = 0;
    virtual void applyModel(cv::Mat& frame) = 0;
};
