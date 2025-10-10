#include "soft_rasterizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::string jsonPath = "infos/face_sculpt.json";
    int H = 256, W = 256;
    if (argc >= 2) jsonPath = argv[1];
    if (argc >= 4) { H = atoi(argv[2]); W = atoi(argv[3]); }

    auto prims = loadPrimitivesFromJson(jsonPath);
    std::cout << "Loaded primitives: " << prims.size() << std::endl;

    cv::Mat imgf = renderSoft(prims, H, W, 100.0);
    // convert to 8-bit BGR for saving
    cv::Mat out8(H, W, CV_8UC3);
    for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
        cv::Vec4d v = imgf.at<cv::Vec4d>(y,x);
        cv::Vec3b pix;
        pix[0] = (unsigned char)std::round(std::clamp(v[0], 0.0, 1.0) * 255.0);
        pix[1] = (unsigned char)std::round(std::clamp(v[1], 0.0, 1.0) * 255.0);
        pix[2] = (unsigned char)std::round(std::clamp(v[2], 0.0, 1.0) * 255.0);
        out8.at<cv::Vec3b>(y,x) = pix;
    }
    cv::imwrite("output_soft_.png", out8);
    std::cout << "Saved output_soft.png" << std::endl;
    return 0;
}
