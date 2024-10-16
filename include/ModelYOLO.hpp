#include "Model.hpp"

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class ModelYOLO : public Model {
public:
    ModelYOLO(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const bool &runWithCuda = true);
    ~ModelYOLO() = default;

    bool isLoaded() override;
    std::vector<Detection> applyModel(const cv::Mat &input) override;
    void drawDetections(cv::Mat &frame, const std::vector<Detection>& detections) override; 

private:
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string mModelPath;
    cv::Size2f mModelInputShape;
    bool mCudaEnabled;

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

    // TODO: Make not hardcoded:
    float mModelConfidenceThreshold {0.25};
    float mModelScoreThreshold      {0.45};
    float mModelNMSThreshold        {0.50};

    cv::dnn::Net mNet;

};
