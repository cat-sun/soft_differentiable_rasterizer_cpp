#include "primitive_render.h"
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>

int main(int argc, char** argv) {
    try {
        // 配置参数
       
        std::string initParamPath = "infos/face_sculpt.json";
        std::string outputDir = "output/20250911/face_sculpt";

        // 初始化优化器配置
        OptimizerConfig config;
        config.lr = 0.001;
        config.decayRate = 0.99;
        config.decaySteps = 100;
        config.maxSteps = 500;
        config.softness = 150.0;
        config.clipNorm = 1.0;

        PrimitiveOptimizer optimizer(config);
        std::filesystem::create_directories(outputDir);

        // 加载目标图像和初始图元
        auto start = std::chrono::high_resolution_clock::now();
        std::string targetPath = "images/face_sculpt.png";
        Mat targetImg = MainUtils::loadTargetImage(targetPath);
        if (targetImg.empty()) {
            std::cerr << "Failed to load target image: " << targetPath << std::endl;
            return 1;
        }

        auto initPrims = MainUtils::loadPrimitivesFromJson(initParamPath);
        std::cout << "Loaded " << initPrims.size() << " primitives:\n";
        for (size_t i = 0; i < initPrims.size(); ++i) {
            if (initPrims[i])
                std::cout << "  [" << i << "] type=" << initPrims[i]->getTypeId() << " color=" << initPrims[i]->color.transpose() << "\n";
            else
                std::cout << "  [" << i << "] null\n";
        }

        // fallback primitive when none loaded
        if (initPrims.empty()) {
            std::cout << "No primitives loaded from JSON; creating a default circle to visualize." << std::endl;
            initPrims.clear();
            initPrims.push_back(std::make_unique<Circle>(Vector2(0.5, 0.5), 0.25, Vector4(0,0,0,1), 1.0));
        }

        int width = targetImg.cols;
        int height = targetImg.rows;

        // 初始渲染
        Mat initialImg = DifferentiableRasterizer::render(initPrims, width, height);
        imwrite(outputDir + "/initial_image.png", initialImg);
        std::cout << "初始图像已保存" << std::endl;

        // 两阶段优化
        auto optimizedPrims = optimizer.stagedOptimize(initPrims, targetImg);

        // 最终渲染与结果
        Mat finalImg = DifferentiableRasterizer::render(optimizedPrims, width, height);
        std::filesystem::create_directories(outputDir);
        cv::imwrite(outputDir + "/final_image.png", finalImg);
        std::cout << "优化完成，最终图像已保存: " << outputDir + "/final_image.png" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Total time: " << elapsed.count() << "s\n";

        return 0;
    } 
    catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }
}
