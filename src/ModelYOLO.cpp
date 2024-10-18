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

#include "ModelYOLO.hpp"

#include <opencv2/opencv.hpp>

static cv::Scalar generateColorFromScalar(int scalar, float nClasses) {
    float H = (scalar / nClasses) * 360.0f;
    float S = 0.75f;  
    float V = 0.75f; 

    float C = V * S;
    float X = C * (1 - std::fabs(fmod(H / 60.0, 2) - 1));
    float m = V - C;
    float r;
    float g;
    float b;

    if (H >= 0 && H < 60) {
        r = C;
        g = X;
        b = 0;
    } 
    else if (H >= 60 && H < 120) {
        r = X;
        g = C;
        b = 0;
    } 
    else if (H >= 120 && H < 180) {
        r = 0;
        g = C;
        b = X;
    } 
    else if (H >= 180 && H < 240) {
        r = 0;
        g = X;
        b = C;
    } 
    else if (H >= 240 && H < 300) {
        r = X;
        g = 0;
        b = C;
    } 
    else {
        r = C;
        g = 0;
        b = X;
    }

    int R = static_cast<int>((r + m) * 255);
    int G = static_cast<int>((g + m) * 255);
    int B = static_cast<int>((b + m) * 255);

    return cv::Scalar(B, G, R);
}

ModelYOLO::ModelYOLO(const std::string& onnxModelPath, const cv::Size& modelInputShape, const bool& runWithCuda,
                     const float confidenceThreshold, const float scoreThreshold, const float NMSThreshold)
: mModelPath(onnxModelPath), mModelInputShape(modelInputShape), mCudaEnabled(runWithCuda), 
  mModelConfidenceThreshold(confidenceThreshold), mModelScoreThreshold(scoreThreshold), 
  mModelNMSThreshold(NMSThreshold)
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

    float xFactor = modelInput.cols / mModelInputShape.width;
    float yFactor = modelInput.rows / mModelInputShape.height;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classesScores = data+4;

            cv::Mat scores(1, mClasses.size(), CV_32FC1, classesScores);
            cv::Point classId;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &classId);

            if (maxClassScore > mModelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                classIds.push_back(classId.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * xFactor);
                int top = int((y - 0.5 * h) * yFactor);

                int width = int(w * xFactor);
                int height = int(h * yFactor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // yolov5
        {
            float confidence = data[4];

            if (confidence >= mModelConfidenceThreshold)
            {
                float *classesScores = data+5;

                cv::Mat scores(1, mClasses.size(), CV_32FC1, classesScores);
                cv::Point classId;
                double maxClassScore;

                minMaxLoc(scores, 0, &maxClassScore, 0, &classId);

                if (maxClassScore > mModelScoreThreshold)
                {
                    confidences.push_back(confidence);
                    classIds.push_back(classId.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * xFactor);
                    int top = int((y - 0.5 * h) * yFactor);

                    int width = int(w * xFactor);
                    int height = int(h * yFactor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, mModelScoreThreshold, mModelNMSThreshold, nmsResult);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nmsResult.size(); ++i)
    {
        int idx = nmsResult[i];

        Detection result;
        result.classId = classIds[idx];
        result.confidence = confidences[idx];

        result.color = generateColorFromScalar(result.classId, mClasses.size());

        result.className = mClasses[result.classId];
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
    int max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(max, max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}