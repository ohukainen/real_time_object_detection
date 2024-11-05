#include "Input.hpp"

#include "ModelYOLO.hpp"

#include <exception>
#include <format>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

struct Args {
    ModelArgs modelArgs;
    int deviceNr = -1;
    std::string videoPath;
    float scalefactor;
    bool rescale = false;
};

static void usage() {
    std::cout << "Usage: " << "real_time_object_detection --model-path <path> --classes-filepath [path] --devive-nr [nr] --video-path [path] --scale-factor [factor] --use-cuda\n" << std::endl;
    std::cout << "Required <>:" << std::endl;
    std::cout << "  --model-path    Path to the detection model.\n" << std::endl;
    std::cout << "Optional []:" << std::endl;
    std::cout << "  --classes-filepath  Path to json file with classnames declared as an array with name classes." << std::endl;
    std::cout << "  --devive-nr         Camera source device nr, can't be provided together with --video-path." << std::endl;
    std::cout << "  --video-path        Path to video file, can't be provided together with --device-nr." << std::endl;
    std::cout << "  --scale-factor      Scale factor for output." << std::endl;
    std::cout << "Flags:" << std::endl;
    std::cout << "  --use-cuda          If flag is provided cuda will be used if possible." << std::endl;
    std::cout << "  --help              Display help.\n" << std::endl;
    std::cout << "Example usage:        real_time_object_detection --model-path C:/models/yolov8n.onnx" << std::endl;
}

static Args parseArgs(int argc, char* argv[]) {
    if (argc > 64) {
        throw std::runtime_error("too many input parameters.");
    }

    Args args;
    const std::vector<std::string_view> inputs(argv + 1, argv + argc);
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        const std::string_view& arg = *it;
        if (arg == "--model-path") {
            if (std::next(it) == inputs.end() || (it + 1)->starts_with("-")) {
                throw std::runtime_error("missing argument for model path.");
            }
            args.modelArgs.modelPath= *++it;
        }
        else if (arg == "--classes-filepath") {
            if (std::next(it) == inputs.end() || (it + 1)->starts_with("-")) {
                throw std::runtime_error("missing argument for classes filepath.");
            }
            args.modelArgs.classesFilepath = *++it;
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
            float scalefactor = std::stof((*++it).data());
            if (scalefactor < 0.1f || scalefactor > 3.0f) {
                throw std::runtime_error("scalefactor has to be between 0.1 and 3.0.");
            }
            args.scalefactor = scalefactor;
            args.rescale = true;
        }
        else if (arg == "--use-cuda") {
            if (std::next(it) == inputs.end()) {
                throw std::runtime_error("missing argument for classes filepath.");
            }
            args.modelArgs.runWithCuda = true;
        }
        else if (arg == "--help") {
            usage(); 
            return Args(); 
        }
        else {
            throw std::runtime_error(std::format("recieved unknown argument not linked to any flag: {}.", arg));

        }
    }

    if (args.modelArgs.modelPath.empty()) {
        throw std::runtime_error("no model path was provided.");
    }

    if (args.deviceNr == -1 && args.videoPath.empty()) {
        args.deviceNr = 0;
    }

    return args;
}

int main(int argc, char* argv[]) {
    Args args;
    try {
        args = parseArgs(argc, argv);
    }
    catch (std::exception& er) {
        std::cerr << "Unable to parse args with message: " << er.what() << std::endl << std::endl;
        usage();
        return -1;
    }

    if (args.modelArgs.modelPath.empty()) {
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
    
    std::unique_ptr<Model> model;
    try {
        model = std::make_unique<ModelYOLO>(args.modelArgs);
    }
    catch (std::exception& er) {
        std::cerr << "Unable to create model with message: " << er.what() << std::endl << std::endl;
        usage();
        return -1;
    }

    if (!model->isLoaded()) {
        std::cerr << "Error: Unable to load the model." << std::endl;
        return -1;
    }

    cv::Mat frame;
    if (!input->getFrame(frame)) {
        std::cerr << "Error: Unable to capture frame." << std::endl;
        return -1;
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

        if (args.rescale) {
            resize(frame, frame, cv::Size(), args.scalefactor, args.scalefactor, cv::INTER_NEAREST);
        }

        cv::imshow("Real-Time Object Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}
