#include <Input.hpp>
#include <opencv2/core/types.hpp>

Input::Input(int device)
: mIsVideo(false)
{
    mCap.open(device);
}

Input::Input(const std::string& filepath)
: mIsVideo(true)
{
    mCap.open(filepath);
}

bool Input::capturing() {
    return mCap.isOpened();
}

bool Input::getFrame(cv::Mat& frame) {
    return mCap.read(frame);
}
