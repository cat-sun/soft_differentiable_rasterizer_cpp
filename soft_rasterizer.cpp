#include "soft_rasterizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>

using json = nlohmann::json;
using namespace std;

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<std::unique_ptr<Primitive>> loadPrimitivesFromJson(const std::string &path) {
    std::vector<std::unique_ptr<Primitive>> out;
    std::ifstream ifs(path);
    if (!ifs) return out;
    json j; ifs >> j;

    auto shapes_id = j["shapes id"];
    auto shapes_op = j["shapes op"];
    auto shapes_params = j["shapes params"];

    size_t n = shapes_id.size();
    std::cerr << "元素个数: " << n << std::endl;
    for (size_t i = 0; i < n; ++i) {
        int prim_type = shapes_id[i].get<int>();
        auto sop = shapes_op[i].get<int>();
        auto sparams = shapes_params[i];
        Eigen::Vector4d color = (sop == 1) ? Eigen::Vector4d(0,0,0,0.999) : Eigen::Vector4d(1,1,1,0.999);
        double depth = double(i+1);
        std::cerr << "参数类型: " << prim_type << std::endl;
        try {
            switch (prim_type) {
                case 0: {
                    double cx = sparams[0].get<double>();
                    double cy = sparams[1].get<double>();
                    double r = sparams[2].get<double>();
                    Vector2 center(cx, cy);
                    out.push_back(std::make_unique<Circle>(center, r, color, depth));
                    break;
                }
                case 1: {
                    double cx = sparams[0].get<double>();
                    double cy = sparams[1].get<double>();
                    double w = sparams[2].get<double>();
                    double h = sparams[3].get<double>();
                    double theta = sparams[4].get<double>();
                    double roundp = sparams[5].get<double>();
                    Vector2 center(cx, cy);
                    Vector2 size(w, h);
                    out.push_back(std::make_unique<Rectangle>(center, size, theta, roundp, color, depth));
                    break;
                }
                case 2: {
                    double cx = sparams[0].get<double>();
                    double cy = sparams[1].get<double>();
                    double r = sparams[2].get<double>();
                    double theta = sparams[3].get<double>();
                    double roundp = sparams[4].get<double>();
                    Vector2 center(cx, cy);
                    out.push_back(std::make_unique<EquilateralTriangle>(center, r, theta, roundp, color, depth));
                    break;
                }
                default: {
                    if (sparams.size() >= 3) {
                        double cx = sparams[0].get<double>();
                        double cy = sparams[1].get<double>();
                        double r = sparams[2].get<double>();
                        Vector2 center(cx, cy);
                        out.push_back(std::make_unique<Circle>(center, r, color, depth));
                    }
                    break;
                }
            }
        } catch (...) {}
    }
    return out;
}

std::vector<Eigen::Vector2d> createNormalizedGrid(int H, int W) {
    std::vector<Eigen::Vector2d> grid; grid.reserve(H*W);
    for (int y = 0; y < H; ++y) {
        double ny = double(y) / double(std::max(1, H-1));
        for (int x = 0; x < W; ++x) {
            double nx = double(x) / double(std::max(1, W-1));
            grid.emplace_back(nx, ny);
        }
    }
    return grid;
}

cv::Mat renderSoft(const std::vector<std::unique_ptr<Primitive>> &prims, int H, int W, double softness) {
    cv::Mat out(H, W, CV_64FC4, cv::Scalar(0,0,0,0)); // double RGBA
    auto grid = createNormalizedGrid(H,W);
    size_t M = prims.size();
    // precompute depths and colors
    std::vector<double> depths(M); std::vector<Vector4> colors(M);
    for (size_t i=0;i<M;++i){ depths[i]=prims[i]->depth; colors[i]=prims[i]->color; }

    // For each pixel compute sdf for every primitive (brute force)
    for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
        size_t idx = y*W + x;
        Eigen::Vector2d pt = grid[idx];
        Vector2 pt2(pt(0), pt(1));
        std::vector<double> sdfv(M);
        for (size_t i=0;i<M;++i) {
            double s = prims[i]->sdf(pt2);
            sdfv[i] = std::clamp(s, -50.0, 50.0);
        }
        // coverage alpha
        std::vector<double> coverage(M);
        for (size_t i=0;i<M;++i) coverage[i] = sigmoid(-sdfv[i]*softness) * colors[i](3);
        // depth weights: use softmax on -depth (closer -> larger weight)
        double maxd = -1e9; for (double d : depths) maxd = std::max(maxd, -d*0.01);
        std::vector<double> expd(M); double sumexp=0.0;
        for (size_t i=0;i<M;++i){ expd[i] = std::exp(-depths[i]*0.01 - maxd); sumexp += expd[i]; }
        std::vector<double> weights(M);
        for (size_t i=0;i<M;++i) weights[i] = (expd[i]/(sumexp+1e-12)) * coverage[i];
        double wsum=1e-12; for (double w:weights) wsum+=w;
        Eigen::Vector3d rgb(0,0,0); double alpha=0.0;
        for (size_t i=0;i<M;++i) {
            rgb += Eigen::Vector3d(colors[i](0),colors[i](1),colors[i](2)) * weights[i] / wsum;
            alpha += coverage[i] * (expd[i]/(sumexp+1e-12));
        }
        out.at<cv::Vec4d>(y,x)[0] = rgb(2); // B
        out.at<cv::Vec4d>(y,x)[1] = rgb(1); // G
        out.at<cv::Vec4d>(y,x)[2] = rgb(0); // R
        out.at<cv::Vec4d>(y,x)[3] = alpha;
    }
    // convert to CV_8UC3 or keep CV_64FC4 caller can convert
    return out;
}


