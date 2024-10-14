#include "ModelONNX.hpp"
#include <opencv2/opencv.hpp>

ModelONNX::ModelONNX(const std::string& modelPath) {
    mNet = cv::dnn::dnn4_v20240521::readNetFromONNX(modelPath);
}

bool ModelONNX::isLoaded() {
    return !mNet.empty();
}

void ModelONNX::applyModel(cv::Mat& frame) {
    double scale = 1.0;
    cv::Size size = cv::Size(640, 640);
    cv::Scalar mean = cv::Scalar(0, 0, 0); 
    bool swapRB = true; 

    std::cout << "frame: " << frame.size()  << std::endl;

    cv::Mat blob = cv::dnn::blobFromImage(frame, scale, size, mean, swapRB, false, CV_32F);

    mNet.setInput(blob);
    cv::Mat output = mNet.forward();

    std::cout << "output: " << output.size() << std::endl;

    // float confidenceThreshold = 0.25;
    // float nmsThreshold = 0.45;        

    // std::vector<cv::Rect> boxes;
    // std::vector<float> confidences;
    // std::vector<int> classIds;

    // // Extract data from the output
    // // YOLOv8 typically outputs [1, N, 85] where N is the number of detections
    // int numDetections = output.size[1];
    // int numAttributes = output.size[2];

    // std::cout << "num detections: " << numDetections << std::endl;
    // std::cout << "num attributes: " << numAttributes << std::endl;

    // float* data = (float*)output.ptr<float>();

    // for (int i = 0; i < numDetections; ++i) {
    //     float confidence = data[4];
    //     if (confidence >= confidenceThreshold) {
    //         std::cout << confidence << std::endl;
    //         // Find the class with the highest score
    //         int classId = -1;
    //         float maxClassScore = 0.0f;
    //         for (int j = 5; j < numAttributes; ++j) {
    //             if (data[j] > maxClassScore) {
    //                 maxClassScore = data[j];
    //                 classId = j - 5;
    //             }
    //         }

    //         if (classId >= 0 && maxClassScore > confidenceThreshold) {
    //             float x_center = data[0] * frame.cols;
    //             float y_center = data[1] * frame.rows;
    //             float width = data[2] * frame.cols;
    //             float height = data[3] * frame.rows;

    //             int left = static_cast<int>(x_center - width / 2);
    //             int top = static_cast<int>(y_center - height / 2);

    //             boxes.emplace_back(cv::Rect(left, top, static_cast<int>(width), static_cast<int>(height)));
    //             confidences.emplace_back(confidence * maxClassScore); // Combined confidence
    //             classIds.emplace_back(classId);
    //         }
    //     }
    //     data += numAttributes;
    // }

    // std::vector<int> indices;
    // cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    // for (size_t i = 0; i < indices.size(); ++i) {
    //     int idx = indices[i];
    //     cv::Rect box = boxes[idx];  
    //     float conf = confidences[idx];
    //     int classId = classIds[idx];

    //     // Ensure bounding box is within the image
    //     box &= cv::Rect(0, 0, frame.cols, frame.rows);

    //     // Draw bounding box
    //     rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
    // }
}
