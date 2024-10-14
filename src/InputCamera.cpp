#include <InputCamera.hpp>

InputCamera::InputCamera(int device) {
    mCap.open(device);
}

bool InputCamera::inputWorking() {
    return mCap.isOpened();
}


bool InputCamera::getFrame(cv::Mat& frame) {
    return mCap.read(frame);
}
