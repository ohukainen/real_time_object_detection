// This file is part of a modified version of the Ultralytics project (https://github.com/ultralytics/ultralytics)
//
// Modified by Johannes Källstad 2024-10-18
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "Model.hpp"

struct Detection
{
    int classId;
    std::string className;
    float confidence;
    cv::Scalar color;
    cv::Rect box;
};

class ModelYOLO : public Model {
public:
    ModelYOLO(const std::string &onnxModelPath, 
              const cv::Size &modelInputShape = {640, 640}, 
              const bool &runWithCuda = false,
              const float confidenceThreshold = 0.25,
              const float scoreThreshold = 0.45,
              const float NMSThreshold = 0.50);
    ~ModelYOLO() = default;

    bool isLoaded() override;
    std::vector<Detection> applyModel(const cv::Mat &input) override;
    void drawDetections(cv::Mat &frame, const std::vector<Detection>& detections) override; 

private:
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string mModelPath;
    cv::Size2f mModelInputShape;
    bool mCudaEnabled;
    float mModelConfidenceThreshold;
    float mModelScoreThreshold;
    float mModelNMSThreshold;

    std::vector<std::string> mClasses{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
                                        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                                        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
                                        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
                                        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
                                        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
                                        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
                                        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
                                        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
                                        "hair drier", "toothbrush"};

    cv::dnn::Net mNet;
};
