#include <camera.hpp>

Camera::Camera(int device, int width, int height) {
    mCap.open(device);

    if (mCap.isOpened()) {
            mCap.set(cv::CAP_PROP_FRAME_WIDTH, width);
            mCap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
}

Camera::~Camera() {
    mCap.release();
}

bool Camera::isOpened() const {
    return mCap.isOpened();
}


bool Camera::getFrame(const cv::Mat& frame) {
    return mCap.read(frame);
}
