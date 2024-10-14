#include <Model.hpp>

class ModelONNX : public Model {
public:
    ModelONNX(const std::string& modelPath);
    ~ModelONNX() = default;

    ModelONNX(const ModelONNX &) = delete;
    ModelONNX(ModelONNX &&) = default;
    ModelONNX &operator=(const ModelONNX &) = delete;
    ModelONNX &operator=(ModelONNX &&) = default;


    bool isLoaded() override;
    void applyModel(cv::Mat& frame) override;

private:
    cv::dnn::Net mNet;

};
