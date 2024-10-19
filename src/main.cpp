#include "Input.hpp"

#include "ModelYOLO.hpp"

#include <exception>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

struct Args {
    std::string modelPath;
    int deviceNr = -1;
    std::string videoPath;
    float scalefactor;
};

static void usage() {
    std::cout << "Usage: " << "real_time_object_detection --model-path <path> --devive-nr [nr] --video-path [path] --scale-factor [factor] \n" << std::endl;
    std::cout << "Required <>:" << std::endl;
    std::cout << "  --model-path    Path to the detection model.\n" << std::endl;
    std::cout << "Optional []:" << std::endl;
    std::cout << "  --devive-nr     Camera source device nr, can't be provided together with --video-path." << std::endl;
    std::cout << "  --video-path    Path to video file, can't be provided together with --device-nr." << std::endl;
    std::cout << "  --scale-factor  Scale factor for output." << std::endl;
    std::cout << "  --help          Display help.\n" << std::endl;
    std::cout << "Example usage:    real_time_object_detection --model-path C:/models/yolov8n.onnx" << std::endl;
}

static Args parseArgs(int argc, char* argv[]) {
    if (argc > 64) {
        throw std::runtime_error("too many input parameters.");
    }

    Args args;
    const std::vector<std::string_view> inputs(argv + 1, argv + argc);
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        const std::string_view& arg = *it;
        if (arg == "--help") {
            usage(); 
            return Args(); 
        }
        else if (arg == "--model-path") {
            if (std::next(it) == inputs.end() || (it + 1)->starts_with("-")) {
                throw std::runtime_error("missing argument for model path.");
            }
            args.modelPath = *++it;
        }
        else if (arg == "--device-nr") {
            if (std::next(it) == inputs.end() || (it + 1)->starts_with("-")) {
                throw std::runtime_error("missing argument for devide nr.");
            }
            else if (!args.videoPath.empty()) {
                throw std::runtime_error("arguments for video path and devide nr was found.");
            }
            args.deviceNr = std::stoi((*++it).data());
        }
        else if (arg == "--video-path") {
            if (std::next(it) == inputs.end() || (it + 1)->starts_with("-")) {
                throw std::runtime_error("missing argument for video path.");
            }
            else if (args.deviceNr != -1) {
                throw std::runtime_error("arguments for video path and devide nr was found.");
            }
            args.videoPath = *++it;
        }
        else if (arg == "--scale-factor") {
            if (std::next(it) == inputs.end() || (it + 1)->starts_with("-")) {
                throw std::runtime_error("missing argument for scalefactor.");
            }
            args.scalefactor = std::stof((*++it).data());
        }
    }

    if (args.modelPath.empty()) {
        throw std::runtime_error("no model path was provided.");
    }

    if (args.deviceNr == -1 && args.videoPath.empty()) {
        args.deviceNr = 0;
    }
    
    if (!args.scalefactor) {
        args.scalefactor = 1.0f;
    }

    return args;
}

int main(int argc, char* argv[]) {
    Args args;
    try {
        args = parseArgs(argc, argv);
    }
    catch (std::exception& er) {
        std::cerr << "Error: Unable to parse args with message: " << er.what() << std::endl << std::endl;
        usage();
        return -1;
    }

    if (args.modelPath.empty()) {
        return 0;
    }

    std::unique_ptr<Input> input;
    if (!args.videoPath.empty()) {
        input = std::make_unique<Input>(args.videoPath); 
    }
    else {
        input = std::make_unique<Input>(args.deviceNr); 
    }

    if (!input->capturing()) {
        std::cerr << "Error: Unable to open input." << std::endl;
        return -1;
    }

    std::unique_ptr<Model> model = std::make_unique<ModelYOLO>(args.modelPath);
    if (!model->isLoaded()) {
        std::cerr << "Error: Unable to load the model." << std::endl;
        return -1;
    }

    cv::Mat frame;
    if (!input->getFrame(frame)) {
        std::cerr << "Error: Unable to capture frame." << std::endl;
        return -1;
    }

    bool rescale = true;
    if (args.scalefactor != 1.0f) {
        rescale = true;
    }

    while (true) { 
        if (!input->getFrame(frame)) {
            if (input->isVideo()) {
                std::cout << "End of video." << std::endl;
                break;
            }
            std::cout << "Error: Unable to capture frame." << std::endl;
            break;
        }
        
        cv::flipND(frame, frame, 1);

        std::vector<Detection> output = model->applyModel(frame);

        model->drawDetections(frame, output);

        if (rescale) {
            resize(frame, frame, cv::Size(), args.scalefactor, args.scalefactor, cv::INTER_NEAREST);
        }

        cv::imshow("Real-Time Object Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}
