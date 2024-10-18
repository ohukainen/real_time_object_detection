#pragma once 
#include <opencv2/opencv.hpp>

struct Detection;

class Model {
public:
    virtual ~Model() = default;
    
    virtual bool isLoaded() = 0;
    virtual std::vector<Detection> applyModel(const cv::Mat &frame) = 0;
    virtual void drawDetections(cv::Mat &frame, const std::vector<Detection>& detections) = 0; 
};
