#include "ModelYOLO.hpp"

#include <opencv2/opencv.hpp>

ModelYOLO::ModelYOLO(const std::string& onnxModelPath, const cv::Size& modelInputShape, const bool& runWithCuda) 
: mModelPath(onnxModelPath), mModelInputShape(modelInputShape), mCudaEnabled(runWithCuda) 
{
    mNet = cv::dnn::readNetFromONNX(mModelPath);
    if (mCudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        mNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        mNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

bool ModelYOLO::isLoaded() {
    return !mNet.empty();
}

std::vector<Detection> ModelYOLO::applyModel(const cv::Mat& input)
{
    cv::Mat modelInput = input;
    if (mModelInputShape.width == mModelInputShape.height) {
        modelInput = formatToSquare(modelInput);
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, mModelInputShape, cv::Scalar(), true, false);
    mNet.setInput(blob);

    std::vector<cv::Mat> outputs;
    mNet.forward(outputs, mNet.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    // TODO: Add logic to infer yolov11
    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / mModelInputShape.width;
    float y_factor = modelInput.rows / mModelInputShape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data+4;

            cv::Mat scores(1, mClasses.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > mModelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // yolov5
        {
            float confidence = data[4];

            if (confidence >= mModelConfidenceThreshold)
            {
                float *classes_scores = data+5;

                cv::Mat scores(1, mClasses.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > mModelScoreThreshold)
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, mModelScoreThreshold, mModelNMSThreshold, nms_result);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        
        // TODO: Decide colors in a more reasonable way
        result.color = cv::Scalar(50, 50, 255);

        result.className = mClasses[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}

void ModelYOLO::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    int detectionsSize = detections.size();

    for (auto & detection : detections)
    {
        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(frame, box, color, 2);

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}

cv::Mat ModelYOLO::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
