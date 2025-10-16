// primitive_render.h - minimal consolidated header
#ifndef PRIMITIVE_RENDER_H
#define PRIMITIVE_RENDER_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <filesystem>
#include <fstream>
#include <algorithm>

// 命名空间简化
namespace fs = std::filesystem;
using json = nlohmann::json;
// Do not import the entire Eigen namespace in a header to avoid name collisions.
// We'll use fully-qualified Eigen types for clarity.
// Avoid `using namespace cv;` in a header to prevent name collisions (e.g. cv::Scalar)
// Import only the specific OpenCV symbols we need.
using cv::Mat;
using cv::Vec4b;
using cv::Point;
using cv::imread;
using cv::imwrite;
using cv::cvtColor;
using cv::Sobel;
using cv::absdiff;
using cv::mean;
using cv::putText;

// 64位精度（匹配JAX的double）
using Scalar = double;
using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
using MatrixX2 = Eigen::Matrix<Scalar, Eigen::Dynamic, 2>;
using MatrixX3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
// 图元基类（纯虚函数定义接口）
class Primitive {
public:
    Vector2 center;       // 归一化中心坐标
    Vector4 color;        // RGBA颜色（0~1）
    Scalar depth;         // 深度值（优化用）

    Primitive(Vector2 c, Vector4 col, Scalar d) 
        : center(c), color(col), depth(d) {}
    virtual ~Primitive() = default;

    // 单个点SDF计算
    virtual Scalar sdf(const Vector2& point) const = 0;
    // 点集SDF批量计算
    virtual VectorX sdf(const MatrixX2& points) const {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }
    // 克隆图元（参数更新用）
    virtual std::unique_ptr<Primitive> clone() const = 0;
    // 获取图元类型ID（序列化用）
    virtual int getTypeId() const = 0;
    // 序列化/反序列化
    virtual void serialize(std::ofstream& os) const {
        os << center.x() << " " << center.y() << " "
           << color.x() << " " << color.y() << " " << color.z() << " " << color.w() << " "
           << depth << " ";
    }
    virtual void deserialize(std::ifstream& is) {
        is >> center.x() >> center.y()
           >> color.x() >> color.y() >> color.z() >> color.w()
           >> depth;
    }
};

// 0. 圆形图元
class Circle : public Primitive {
public:
    Scalar radius;  // 归一化半径

    Circle(Vector2 c, Scalar r, Vector4 col, Scalar d) 
        : Primitive(c, col, d), radius(r) {}

    // (legacy constructor removed) use Eigen Vector2/Vector4 based constructor

    Scalar sdf(const Vector2& point) const override {
        return (point - center).norm() - radius;
    }

    VectorX sdf(const MatrixX2& points) const override {
        MatrixX2 diff = points.rowwise() - center.transpose();
        VectorX res = diff.rowwise().norm();
        return (res.array() - radius).matrix();
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Circle>(center, radius, color, depth);
    }

    int getTypeId() const override { return 0; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << radius << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> radius;
    }
};

// 1. 矩形图元
class Rectangle : public Primitive {
public:
    Vector2 size;         // 宽高（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）
    Scalar roundParam;    // 圆角半径（归一化）

    Rectangle(Vector2 c, Vector2 s, Scalar rt, Scalar rp, Vector4 col, Scalar d) 
        : Primitive(c, col, d), size(s), rotateTheta(rt), roundParam(rp) {}

    Scalar sdf(const Vector2& point) const override {
        // 坐标平移+反向旋转
        Vector2 translated = point - center;
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMat;
        rotMat << cosT, sinT, -sinT, cosT;
        Vector2 rotated = rotMat * translated;

        // 圆角矩形SDF计算
        Vector2 halfSize = size / 2;
        Vector2 p = rotated.cwiseAbs() - halfSize + Vector2(roundParam, roundParam);
        Vector2 q = p.cwiseMax(0.0);
        Scalar outside = q.norm();
        Scalar inside = p.maxCoeff();
        inside = std::min(inside, 0.0);
        return outside + inside - roundParam;
    }

    VectorX sdf(const MatrixX2& points) const override {
        MatrixX2 translated = points.rowwise() - center.transpose();
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMat;
        rotMat << cosT, sinT, -sinT, cosT;
        MatrixX2 rotated = translated * rotMat;

        VectorX p0 = rotated.col(0).cwiseAbs().array() - size.x()/2 + roundParam;
        VectorX p1 = rotated.col(1).cwiseAbs().array() - size.y()/2 + roundParam;
        VectorX q0 = p0.cwiseMax(0.0), q1 = p1.cwiseMax(0.0);
        VectorX outside = (q0.array().square() + q1.array().square()).sqrt().matrix();
        VectorX inside = p0.cwiseMax(p1).cwiseMin(0.0);
        return (outside + inside).array() - roundParam;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Rectangle>(center, size, rotateTheta, roundParam, color, depth);
    }

    int getTypeId() const override { return 1; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << size.x() << " " << size.y() << " " << rotateTheta << " " << roundParam << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> size.x() >> size.y() >> rotateTheta >> roundParam;
    }
};

// 2. 等边三角形图元
class EquilateralTriangle : public Primitive {
public:
    Scalar radius;        // 外接圆半径（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）
    Scalar roundParam;    // 圆角半径（归一化）

    EquilateralTriangle(Vector2 c, Scalar r, Scalar rt, Scalar rp, Vector4 col, Scalar d) 
        : Primitive(c, col, d), radius(r), rotateTheta(rt), roundParam(rp) {}

    Scalar sdf(const Vector2& point) const override {
        Scalar length = 2 * radius * cos(M_PI / 6);  // 边长=2*R*cos(30°)
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMat;
        rotMat << cosT, -sinT, sinT, cosT;
        Vector2 translated = point - center;
        Vector2 rotated = rotMat * translated;

        // 对称性处理
        Scalar x = abs(rotated.x()), y = rotated.y();
        Scalar k = sqrt(3.0);
        Scalar rHalf = length / 2;

        // 区域判断与SDF计算
        Scalar pp0 = x, pp1 = y;
        if (x + k * y > 0) {
            pp0 = (x - k * y) / 2;
            pp1 = (-k * x - y) / 2;
        }
        pp0 -= rHalf;
        pp1 += rHalf / k;
        pp0 = std::max(pp0, -length);
        pp0 = std::min(pp0, 0.0);
        pp0 = x - pp0;

        Scalar sdf = -sqrt(pp0*pp0 + pp1*pp1) * (pp1 < 0 ? -1 : 1);
        return sdf - roundParam;
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<EquilateralTriangle>(center, radius, rotateTheta, roundParam, color, depth);
    }

    int getTypeId() const override { return 2; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << radius << " " << rotateTheta << " " << roundParam << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> radius >> rotateTheta >> roundParam;
    }
};

// 3. 弯曲胶囊
class CurvedCapsule : public Primitive {
public:
    Scalar length;       // 长度
    Scalar a;            // 曲率参数
    Scalar rotateTheta;  // 旋转角度（弧度）
    Scalar roundParam;   // 圆角参数

    // 构造函数
    CurvedCapsule(Vector2 c, Scalar l, Scalar a_param, Scalar rt, Scalar rp, 
                 Vector4 col, Scalar d) 
        : Primitive(c, col, d), length(l), a(a_param), rotateTheta(rt), roundParam(rp) {}

    // 单个点的SDF计算
    Scalar sdf(const Vector2& point) const override {
        const Scalar cx = center.x();
        const Scalar cy = center.y();
        const Scalar eps1 = 1e-6;
        
        // 角度修正
        Scalar theta = fmod(rotateTheta - M_PI / 2, 2 * M_PI);
        if (theta < 0) theta += 2 * M_PI;

        // 坐标变换
        Scalar cos_theta = cos(theta);
        Scalar sin_theta = sin(theta);
        
        // 旋转矩阵转置乘以(点 - 中心点)
        Scalar px = cos_theta * (point.x() - cx) + sin_theta * (point.y() - cy);
        Scalar py = -sin_theta * (point.x() - cx) + cos_theta * (point.y() - cy);

        // 线段SDF计算
        Scalar clamped_x = std::clamp(px, -length, length);
        Scalar d_line = sqrt(pow(px - clamped_x, 2) + pow(py, 2) + 1e-12);

        // 曲率方向和大小
        Scalar abs_a = sqrt(a * a + eps1);
        Scalar sign_a = (a >= 0) ? 1.0 : -1.0;

        // 圆弧参数
        Scalar scx = cos(abs_a / 2.0);
        Scalar scy = sin(abs_a / 2.0);
        Scalar ra = std::clamp(length / abs_a, 0.0, 1000.0);  // 半径

        // 圆弧中心下移（方向依赖sign_a）
        Scalar py2 = py - sign_a * ra;
        Scalar px2 = fabs(px);

        // 反射计算
        Scalar dot_sp = scx * px2 + sign_a * scy * py2;
    Scalar m = std::clamp(dot_sp, 0.0, std::numeric_limits<Scalar>::infinity());
        Scalar qx = px2 - 2.0 * scx * m;
        Scalar qy = py2 - 2.0 * sign_a * scy * m;

        // 圆弧距离
        Scalar qlen = sqrt(qx * qx + qy * qy);
        Scalar u = fabs(ra) - qlen;
        Scalar d_arc_left = sqrt(qx * qx + pow(qy + sign_a * ra, 2) + 1e-12);
        Scalar d_arc = (qx < 0.0) ? d_arc_left : fabs(u);

        // 平滑过渡权重
        Scalar a_abs = fabs(a);
        Scalar curvature_based_eps = 0.1 * (length / (roundParam + 1e-8));
        curvature_based_eps = std::clamp(curvature_based_eps, 1e-3, 1e-1);
        Scalar transition_center = 1e-3;  // 固定很小的值
        Scalar transition_width = 5e-4;
        
        Scalar x = (a_abs - transition_center) / (transition_width + 1e-8);
        Scalar k = 0.5 * (tanh(x) + 1.0);
        k = std::clamp(k, 0.0, 1.0);

        // 融合线段和圆弧
        Scalar d = (1.0 - k) * d_line + k * d_arc;
        return d - roundParam;
    }

    // 多点的SDF计算
    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); ++i) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    // 克隆函数
    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<CurvedCapsule>(
            center, length, a, rotateTheta, roundParam, color, depth
        );
    }

    // 类型ID
    int getTypeId() const override { return 3; }  

    // 序列化
    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << length << " " << a << " " << rotateTheta << " " << roundParam << "\n";
    }

    // 反序列化
    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> length >> a >> rotateTheta >> roundParam;
    }
};

// 4. 圆弧
class Arc : public Primitive {
public:
    Scalar shapeTheta;       // 圆弧张角（弧度）
    Scalar rotateTheta;      // 圆弧旋转角度（弧度）
    Scalar radius;           // 圆弧半径
    Scalar roundParamRadius; // 端点圆角半径

    // 构造函数
    Arc(Vector2 c, Scalar st, Scalar rt, Scalar r, Scalar rpr, 
        Vector4 col, Scalar d) 
        : Primitive(c, col, d), 
          shapeTheta(st), 
          rotateTheta(rt), 
          radius(r), 
          roundParamRadius(rpr) {}

    // 单个点的SDF计算
    Scalar sdf(const Vector2& point) const override {
        const Scalar x = center.x();
        const Scalar y = center.y();
        const Scalar eps = 1e-8;
        const Scalar eps_small = 1e-12;

        // 处理半径参数，避免数值问题
        Scalar ra = std::max(radius, eps);
        Scalar rb = std::max(roundParamRadius, 0.0);

        // 构建旋转矩阵的转置（用于反向旋转）
        Scalar cos_rot = std::cos(rotateTheta);
        Scalar sin_rot = std::sin(rotateTheta);
        Eigen::Matrix2d rotate_mat_t;
        rotate_mat_t << cos_rot,  sin_rot,
                       -sin_rot, cos_rot;

        // 平移向量（中心坐标）
        Eigen::Vector2d translation(x, y);

        // 坐标变换：先旋转再平移（转换到局部坐标系）
        Eigen::Vector2d point_vec = point;
        Eigen::Vector2d new_coords = rotate_mat_t * point_vec - rotate_mat_t * translation;

        // 提取局部坐标系下的坐标分量（x0取绝对值，对称处理）
        Scalar coords0 = std::abs(new_coords.x());
        Scalar coords1 = new_coords.y();

        // 计算张角对应的正弦和余弦
        Scalar sin_theta = std::sin(shapeTheta);
        Scalar cos_theta = std::cos(shapeTheta);

        // 判断点是否在圆弧的角度范围内（mask为true时使用圆角SDF）
        bool mask = (cos_theta * coords0 - sin_theta * coords1) > 1e-8;

        // 圆弧主体SDF：到圆心的距离与半径的差，再减去宽度
        Scalar dist_center = std::sqrt(coords0 * coords0 + coords1 * coords1 + eps_small);
        Scalar d1 = std::abs(dist_center - ra) - rb;

        // 圆弧端点圆角SDF：以圆弧终点为圆心的圆
        Eigen::Vector2d arc_center(ra * sin_theta, ra * cos_theta);
        Scalar dx = coords0 - arc_center.x();
        Scalar dy = coords1 - arc_center.y();
        Scalar dist_round_param = std::sqrt(dx * dx + dy * dy + eps_small);
        Scalar d2 = dist_round_param - rb;

        // 组合SDF：根据mask选择主体或圆角SDF
        return mask ? d2 : d1;
    }

    // 多点的SDF计算
    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); ++i) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    // 克隆函数
    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Arc>(
            center, shapeTheta, rotateTheta, radius, roundParamRadius, color, depth
        );
    }

    // 类型ID（确保与其他图元不同）
    int getTypeId() const override { return 4; }

    // 序列化
    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << shapeTheta << " " << rotateTheta << " " 
           << radius << " " << roundParamRadius << "\n";
    }

    // 反序列化
    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> shapeTheta >> rotateTheta >> radius >> roundParamRadius;
    }
};

// 5. 梯形图元
class Trapezoid : public Primitive {
public:
    Scalar width1;        // 底宽（归一化）
    Scalar width2;        // 顶宽（归一化）
    Scalar height;       // 高度（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）
    Scalar roundParam;    // 圆角半径（归一化）

    Trapezoid(Vector2 c, Scalar w1, Scalar w2, Scalar h, Scalar rt, Scalar rp, Vector4 col, Scalar d) 
        : Primitive(c, col, d), width1(w1), width2(w2), height(h), rotateTheta(rt), roundParam(rp) {}

    Scalar sdf(const Vector2& point) const override {
        // 参数约束
        Scalar w1 = std::max(width1, 1e-3);
        Scalar w2 = std::max(width2, 1e-3);
        Scalar h = std::max(height, 1e-3);
        w1 = std::max(w1, w2 + 0.05);

        // 归一化尺度
        Scalar scale = sqrt(w1 * h);
        Scalar r1 = (w1 / 2) / scale, r2 = (w2 / 2) / scale;
        Scalar hHalf = (h / 2) / scale;
        Scalar radius = roundParam / scale;

        // 坐标变换
        Vector2 translated = point - center;
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMat;
        rotMat << cosT, -sinT, sinT, cosT;
        Vector2 p = (rotMat * translated) / scale;

        Scalar x = abs(p.x()), y = p.y();

        // 上下边界SDF
        bool upperMask = (y <= 0);
        Scalar qp1x = upperMask ? std::max(x - r2, 0.0) : std::max(x - r1, 0.0);
        Scalar qp1y = abs(y) - hHalf;
        Scalar d1 = sqrt(qp1x*qp1x + qp1y*qp1y);

        // 斜边SDF
        Scalar apX = x - r1, apY = y - hHalf;
        Scalar abX = r2 - r1, abY = -2 * hHalf;
        Scalar abLen2 = abX*abX + abY*abY + 1e-8;
        Scalar t = std::clamp((apX*abX + apY*abY) / abLen2, 0.0, 1.0);
        Scalar projX = abX * t, projY = abY * t;
        Scalar qp2x = apX - projX, qp2y = apY - projY;
        Scalar d2 = sqrt(qp2x*qp2x + qp2y*qp2y);

        // 内部判断
        bool inside = (qp1y < 0) && (qp2x < 0);
        Scalar sign = inside ? -1.0 : 1.0;
        Scalar d = sign * std::min(d1, d2);

        return (d - radius) * scale;  // 还原尺度
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Trapezoid>(center, width1, width2, height, rotateTheta, roundParam, color, depth);
    }

    int getTypeId() const override { return 5; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << width1 << " " << width2 << " " << height << " " << rotateTheta << " " << roundParam << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> width1 >> width2 >> height >> rotateTheta >> roundParam;
    }
};

// 6. 星形图元（五角星）
class Star : public Primitive {
public:
    Scalar radius;        // 顶点半径（归一化）
    Scalar theta;         // 旋转角度（弧度）
    Scalar externalAngle; // 边角角度（控制尖锐度）
    Scalar roundParam;    // 圆角半径（归一化）
    Scalar k;             // 凹陷参数

    Star(Vector2 c, Scalar r, Scalar t, Scalar ea, Scalar rp, Scalar kVal, Vector4 col, Scalar d) 
        : Primitive(c, col, d), radius(r), theta(t), externalAngle(ea), roundParam(rp), k(kVal) {}

    Scalar sdf(const Vector2& point) const override {
        Scalar r = std::max(radius, 1e-8);
        Scalar rp = std::max(roundParam, 0.0);
        Scalar kVal = std::max(k, 1e-8);
        const Scalar n = 5.0;  // 五角星固定5个角

        // 坐标变换
        Vector2 translated = point - center;
        Scalar cosT = cos(-theta), sinT = sin(-theta);
        Matrix2 rotMat;
        rotMat << cosT, -sinT, sinT, cosT;
        Vector2 p = rotMat * translated;

        // 对称性处理
        p = Vector2(abs(p.x()), p.y());

        // 角度计算
        Scalar angle = atan2(p.x(), p.y());
        Scalar bn = fmod(angle + 2*M_PI, 2*(M_PI/n)) - (M_PI/n);
        Scalar q = abs(sin(bn));

        // 局部坐标转换
        Scalar lenP = p.norm();
        Scalar px = lenP * cos(bn);
        Scalar qClamped = std::clamp(q, 0.0, kVal);
        Scalar py = lenP * (q + 0.5 * pow(std::max(kVal - qClamped, 0.0), 2) / kVal);

        // 线段SDF
        Scalar m = n + externalAngle * (2 - n);
        Scalar an = M_PI / n, en = M_PI / m;
        Vector2 racs(r * cos(an), r * sin(an));
        Vector2 ecs(cos(en), sin(en));

        Vector2 pLocal(px, py);
        pLocal -= racs;
        Scalar dot = ecs.x() * pLocal.x() + ecs.y() * pLocal.y();
        Scalar maxVal = racs.y() / ecs.y();
        Scalar clamped = std::clamp(-dot, 0.0, maxVal);
        pLocal += ecs * clamped;

        Scalar sdf = pLocal.norm() * (pLocal.x() < 0 ? -1 : 1);
        return sdf - rp;
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Star>(center, radius, theta, externalAngle, roundParam, k, color, depth);
    }

    int getTypeId() const override { return 6; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << radius << " " << theta << " " << externalAngle << " " << roundParam << " " << k << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> radius >> theta >> externalAngle >> roundParam >> k;
    }
};

// 7. 半圆图元
class HalfCircle : public Primitive {
public:
    Scalar radius;        // 半径（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）
    Scalar roundParam;    // 圆角半径（归一化）

    HalfCircle(Vector2 c, Scalar r, Scalar rt, Scalar rp, Vector4 col, Scalar d) 
        : Primitive(c, col, d), radius(r), rotateTheta(rt), roundParam(rp) {}

    Scalar sdf(const Vector2& point) const override {
        Scalar isVal = (radius > 1e-6) ? 1.0 : 0.0;
        if (isVal < 1e-6) return 1e10;

        // 坐标变换
        Vector2 translated = point - center;
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Scalar xRot = translated.x() * cosT + translated.y() * sinT;
        Scalar yRot = -translated.x() * sinT + translated.y() * cosT;

        // 半圆SDF计算
        Scalar distCenter = sqrt(xRot*xRot + yRot*yRot + 1e-12);
        Scalar diskSdf = distCenter - radius;
        Scalar lineSdf = xRot;  // 切割线x=0
        Scalar combinedSdf = std::max(diskSdf, -lineSdf);

        // 切割线超出处理
        Scalar absCut = abs(0.0);  // cut_offset=0（右半圆）
        if (absCut > radius) {
            return (0.0 < 0) ? diskSdf : 1e10;
        }
        return combinedSdf - roundParam;
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<HalfCircle>(center, radius, rotateTheta, roundParam, color, depth);
    }

    int getTypeId() const override { return 7; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << radius << " " << rotateTheta << " " << roundParam << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> radius >> rotateTheta >> roundParam;
    }
};

// 8. 等腰三角形图元
class IsoscelesTriangle : public Primitive {
public:
    Scalar width;         // 底宽（归一化）
    Scalar height;        // 高度（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）
    Scalar roundParam;    // 圆角半径（归一化）

    IsoscelesTriangle(Vector2 c, Scalar w, Scalar h, Scalar rt, Scalar rp, Vector4 col, Scalar d) 
        : Primitive(c, col, d), width(w), height(h), rotateTheta(rt), roundParam(rp) {}

    Scalar sdf(const Vector2& point) const override {
        // 坐标变换
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMat;
        rotMat << cosT, -sinT, sinT, cosT;
        Vector2 translated = point - center;
        Vector2 p = rotMat * translated;

        // 顶点坐标
        Vector2 pointV(width / 2, -height);

        // 对称性处理
        p.x() = abs(p.x());

        // 向量计算
        Vector2 pb = pointV - p;
        Vector2 pp = p - pointV;
        Scalar dotPB = pb.x() * pp.x() + pb.y() * pp.y();
        Scalar dotBB = pb.x() * pb.x() + pb.y() * pb.y() + 1e-8;
        Scalar t = std::clamp(dotPB / dotBB, 0.0, 1.0);

        // SDF计算
        Vector2 qp1 = p - (pointV + pb * t);
        Vector2 qp2 = p - Vector2(std::clamp(p.x() / pointV.x(), 0.0, 1.0) * pointV.x(), pointV.y());
        Scalar d1 = qp1.norm(), d2 = qp2.norm();
        Scalar sdf = -std::min(d1, d2) * (p.y() < pointV.y() ? -1 : 1);

        return sdf - roundParam;
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<IsoscelesTriangle>(center, width, height, rotateTheta, roundParam, color, depth);
    }

    int getTypeId() const override { return 8; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << width << " " << height << " " << rotateTheta << " " << roundParam << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> width >> height >> rotateTheta >> roundParam;
    }
};

// 9. 心形图元
class Heart : public Primitive {
public:
    Scalar size;          // 尺度（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）

    Heart(Vector2 c, Scalar s, Scalar rt, Vector4 col, Scalar d) 
        : Primitive(c, col, d), size(s), rotateTheta(rt) {}

    Scalar sdf(const Vector2& point) const override {
        // 坐标变换
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMatT;
        rotMatT << cosT, sinT, -sinT, cosT;
        Vector2 translated = point - center;
        Vector2 p = rotMatT * translated;

        // 单位心形SDF
        p /= size;
        Scalar x = abs(p.x()), y = p.y();

        bool cond = (y + x) > 0.5;
        Scalar dx1 = x - 0.25, dy1 = y - 0.25;
        Scalar d1 = sqrt(dx1*dx1 + dy1*dy1) - (sqrt(2.0)/4.0);

        Scalar dx2a = x, dy2a = y + 0.5;
        Scalar dot2a = dx2a*dx2a + dy2a*dy2a;
        Scalar m = 0.5 * std::max(x + y + 0.5, 0.0);
        Scalar dx2b = x - m, dy2b = y - m + 0.5;
        Scalar dot2b = dx2b*dx2b + dy2b*dy2b;
        Scalar d2 = sqrt(std::min(dot2a, dot2b)) * (x - y - 0.5 < 0 ? -1 : 1);

        Scalar dUnit = cond ? d1 : d2;
        return dUnit * size;
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Heart>(center, size, rotateTheta, color, depth);
    }

    int getTypeId() const override { return 9; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << size << " " << rotateTheta << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> size >> rotateTheta;
    }
};

// 10. 六边形图元
class Hexagon : public Primitive {
public:
    Scalar radius;        // 外接圆半径（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）
    Scalar roundParam;    // 圆角半径（归一化）

    Hexagon(Vector2 c, Scalar r, Scalar rt, Scalar rp, Vector4 col, Scalar d) 
        : Primitive(c, col, d), radius(r), rotateTheta(rt), roundParam(rp) {}

    Scalar sdf(const Vector2& point) const override {
        // 坐标变换
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMatT;
        rotMatT << cosT, sinT, -sinT, cosT;
        Vector2 translated = point - center;
        Vector2 p = rotMatT * translated;

        // 半径调整
        Scalar r = radius * (sqrt(3.0) / 2);
        Vector3 k(-0.866025404, 0.5, 0.577350269);
        p = p.cwiseAbs();

        // 六边形SDF计算
        Scalar dot = k.x() * p.x() + k.y() * p.y();
        p.x() -= 2 * std::min(dot, 0.0) * k.x();
        p.y() -= 2 * std::min(dot, 0.0) * k.y();
        p.x() = std::clamp(p.x(), -k.z() * r, k.z() * r);
        p.y() -= r;

        Scalar sdf = p.norm() * (p.y() < 0 ? -1 : 1);
        return sdf - roundParam;
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Hexagon>(center, radius, rotateTheta, roundParam, color, depth);
    }

    int getTypeId() const override { return 10; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << radius << " " << rotateTheta << " " << roundParam << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> radius >> rotateTheta >> roundParam;
    }
};

// 11. 胶囊图元
class Capsule : public Primitive {
public:
    Scalar radius;        // 半径（归一化）
    Scalar rotateTheta;   // 旋转角度（弧度）
    Scalar roundParam;    // 宽度（归一化）

    Capsule(Vector2 c, Scalar r, Scalar rt, Scalar rp, Vector4 col, Scalar d) 
        : Primitive(c, col, d), radius(r), rotateTheta(rt), roundParam(rp) {}

    Scalar sdf(const Vector2& point) const override {
        Scalar r = std::max(radius, 1e-8);
        Scalar length = 2 * roundParam;

        // 坐标变换
        Scalar cosT = cos(rotateTheta), sinT = sin(rotateTheta);
        Matrix2 rotMat;
        rotMat << cosT, -sinT, sinT, cosT;

        // 端点计算
        Vector2 endpointsLocal[2] = {
            Vector2(0, -length/2),
            Vector2(0, length/2)
        };
        Vector2 a = rotMat * endpointsLocal[0] + center;
        Vector2 b = rotMat * endpointsLocal[1] + center;

        // 线段SDF
        Vector2 pa = point - a;
        Vector2 ba = b - a;
        Scalar baLen2 = ba.dot(ba) + 1e-8;
        Scalar t = std::clamp(pa.dot(ba) / baLen2, 0.0, 1.0);
        Vector2 closest = a + ba * t;
        Scalar dist = (point - closest).norm();

        return dist - r;
    }

    VectorX sdf(const MatrixX2& points) const override {
        VectorX res(points.rows());
        for (int i = 0; i < points.rows(); i++) {
            res(i) = sdf(Vector2(points(i, 0), points(i, 1)));
        }
        return res;
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Capsule>(center, radius, rotateTheta, roundParam, color, depth);
    }

    int getTypeId() const override { return 11; }

    void serialize(std::ofstream& os) const override {
        os << getTypeId() << " ";
        Primitive::serialize(os);
        os << radius << " " << rotateTheta << " " << roundParam << "\n";
    }

    void deserialize(std::ifstream& is) override {
        Primitive::deserialize(is);
        is >> radius >> rotateTheta >> roundParam;
    }
};
class MainUtils {
public:
    static Mat loadTargetImage(const std::string& path) {
        Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) throw std::runtime_error("无法加载目标图像: " + path);
        if (img.channels()==3) cv::cvtColor(img,img,cv::COLOR_BGR2RGBA);
        if (img.channels()==1) cv::cvtColor(img,img,cv::COLOR_GRAY2RGBA);
        return img;
    }
    static std::vector<std::unique_ptr<Primitive>> loadPrimitivesFromJson(const std::string& jsonPath) {
        std::vector<std::unique_ptr<Primitive>> primitives;
        std::ifstream f(jsonPath);
        if (!f.is_open()) return primitives;
        nlohmann::json data; f >> data;
        if (!data.contains("shapes id") || !data.contains("shapes params") || !data.contains("shapes op")) return primitives;
        auto shapes_id = data["shapes id"].get<std::vector<int>>();
        auto shapes_op = data["shapes op"].get<std::vector<int>>();
        auto shapes_params = data["shapes params"].get<std::vector<std::vector<double>>>();
        size_t n = std::min({shapes_id.size(), shapes_op.size(), shapes_params.size()});
        // 逐个加载图元
        for (size_t i = 0; i < shapes_id.size(); ++i) {
            int prim_type = shapes_id[i];
            int sop = shapes_op[i];
            auto& sparams = shapes_params[i];
            
            std::cout << "元素类型 " << prim_type<<  std::endl;

            // 根据操作类型确定颜色
            Vector4 color;
            if (sop == 1) {  // 减集 - 黑色
                color = Vector4(0.0, 0.0, 0.0, 0.999);
            } else {  // 并集(0)或交集(2) - 白色
                color = Vector4(1.0, 1.0, 1.0, 0.999);
            }
            
            // 确保颜色在有效范围内
            color.x() = std::clamp(color.x(), 0.001, 0.999);
            color.y() = std::clamp(color.y(), 0.001, 0.999);
            color.z() = std::clamp(color.z(), 0.001, 0.999);
            
            // 深度按顺序从1开始
            Scalar depth = i + 1;
            
            // 根据图元类型创建相应对象
            switch (prim_type) {
                case 0: {  // 圆形
                    if (sparams.size() < 3) {
                        throw std::runtime_error("圆形参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar radius = sparams[2];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<Circle>(
                        center, radius, color, depth
                    ));
                    break;
                }
                
                case 1: {  // 矩形
                    if (sparams.size() < 6) {
                        throw std::runtime_error("矩形参数不足");
                    }
                    Scalar center_x = std::clamp(sparams[0], 0.0, 1.0);
                    Scalar center_y = std::clamp(sparams[1], 0.0, 1.0);
                    Scalar width = std::max(sparams[2], 0.0);
                    Scalar height = std::max(sparams[3], 0.0);
                    Scalar rotate_theta = sparams[4];
                    Scalar round_param = sparams[5];
                    
                    Vector2 center(center_x, center_y);
                    Vector2 size(width, height);
                    primitives.push_back(std::make_unique<Rectangle>(
                        center, size, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 2: {  // 等边三角形
                    if (sparams.size() < 5) {
                        throw std::runtime_error("等边三角形参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar radius = sparams[2];
                    Scalar rotate_theta = sparams[3];
                    Scalar round_param = sparams[4];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<EquilateralTriangle>(
                        center, radius, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 3: {  // 弯曲胶囊体
                    if (sparams.size() < 6) {
                        throw std::runtime_error("弯曲胶囊体参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar length = sparams[2];
                    Scalar a = sparams[3];
                    Scalar rotate_theta = sparams[4];
                    Scalar round_param = sparams[5];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<CurvedCapsule>(
                        center, length, a, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 4: {  // 圆弧
                    if (sparams.size() < 6) {
                        throw std::runtime_error("圆弧参数不足");
                    }
                    Scalar center_x = std::clamp(sparams[0], 0.0, 1.0);
                    Scalar center_y = std::clamp(sparams[1], 0.0, 1.0);
                    Scalar shape_theta = sparams[2];
                    Scalar rotate_theta = sparams[3];
                    Scalar radius = sparams[4];
                    Scalar round_param = sparams[5];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<Arc>(
                        center, shape_theta, rotate_theta, radius, round_param, color, depth
                    ));
                    break;
                }
                
                case 5: {  // 梯形
                    if (sparams.size() < 7) {
                        throw std::runtime_error("梯形参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar width1 = sparams[2];
                    Scalar width2 = sparams[3];
                    Scalar height = sparams[4];
                    Scalar rotate_theta = sparams[5];
                    Scalar round_param = sparams[6];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<Trapezoid>(
                        center, width1, width2, height, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 6: {  // 星形
                    if (sparams.size() < 7) {
                        throw std::runtime_error("星形参数不足");
                    }
                    Scalar center_x = std::clamp(sparams[0], 0.0, 1.0);
                    Scalar center_y = std::clamp(sparams[1], 0.0, 1.0);
                    Scalar radius = sparams[2];
                    Scalar theta = sparams[3];
                    Scalar external_angle = sparams[4];
                    Scalar rotate_theta = sparams[5];
                    Scalar round_param = sparams[6];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<Star>(
                        center, radius, theta, external_angle, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 7: {  // 半圆形
                    if (sparams.size() < 5) {
                        throw std::runtime_error("半圆形参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar radius = sparams[2];
                    Scalar rotate_theta = sparams[3];
                    Scalar round_param = sparams[4];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<HalfCircle>(
                        center, radius, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 8: {  // 等腰三角形
                    if (sparams.size() < 6) {
                        throw std::runtime_error("等腰三角形参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar width = sparams[2];
                    Scalar height = sparams[3];
                    Scalar rotate_theta = sparams[4];
                    Scalar round_param = sparams[5];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<IsoscelesTriangle>(
                        center, width, height, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 9: {  // 心形
                    if (sparams.size() < 3) {
                        throw std::runtime_error("心形参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar size = sparams[2];
                    Scalar rotate_theta = sparams[3];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<Heart>(
                        center, size, rotate_theta, color, depth
                    ));
                    break;
                }
                
                case 10: {  // 六边形
                    if (sparams.size() < 5) {
                        throw std::runtime_error("六边形参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar radius = sparams[2];
                    Scalar rotate_theta = sparams[3];
                    Scalar round_param = sparams[4];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<Hexagon>(
                        center, radius, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                case 11: {  // 胶囊体
                    if (sparams.size() < 5) {
                        throw std::runtime_error("胶囊体参数不足");
                    }
                    Scalar center_x = sparams[0];
                    Scalar center_y = sparams[1];
                    Scalar radius = sparams[2];
                    Scalar rotate_theta = sparams[3];
                    Scalar round_param = sparams[4];
                    
                    Vector2 center(center_x, center_y);
                    primitives.push_back(std::make_unique<Capsule>(
                        center, radius, rotate_theta, round_param, color, depth
                    ));
                    break;
                }
                
                default:
                    throw std::runtime_error("未知的图元类型: " + std::to_string(prim_type));
            }
        }
        
        return primitives;
    }
};

// Minimal differentiable rasterizer sufficient for tests
class DifferentiableRasterizer {
public:
    static Mat render(const std::vector<std::unique_ptr<Primitive>>& prims, int width, int height, Scalar softness = 150.0) {
        Mat img(height, width, CV_8UC4, cv::Scalar(0,0,0,255));
        if (prims.empty()) return img;
        // naive hard rasterization: draw filled circles for Circle primitives
        for (const auto& p : prims) {
            if (p->getTypeId() == 0) {
                const Circle* c = static_cast<const Circle*>(p.get());
                int cx = (int)std::round(c->center.x() * (width-1));
                int cy = (int)std::round(c->center.y() * (height-1));
                int rad = std::max(1, (int)std::round(c->radius * std::max(width,height)));
                cv::Scalar col(c->color.z()*255, c->color.y()*255, c->color.x()*255, c->color.w()*255);
                cv::circle(img, cv::Point(cx,cy), rad, col, -1, cv::LINE_AA);
            }
            else if (p->getTypeId() == 1){
                const Rectangle* r = static_cast<const Rectangle*>(p.get());
                int cx = (int)std::round(r->center.x() * (width-1));
                int cy = (int)std::round(r->center.y() * (height-1));
                int w = std::max(1, (int)std::round(r->size.x() * (width-1)));
                int h = std::max(1, (int)std::round(r->size.y() * (height-1)));
                cv::Scalar col(r->color.z()*255, r->color.y()*255, r->color.x()*255, r->color.w()*255);
                cv::RotatedRect box(cv::Point(cx,cy), cv::Size(w,h), r->rotateTheta * 180.0 / M_PI);
                cv::Point2f vertices[4];
                box.points(vertices);
                std::vector<cv::Point> pts;
                for (int i=0;i<4;i++) pts.push_back(vertices[i]);
                cv::fillConvexPoly(img, pts, col, cv::LINE_AA);
            }
            else if(p->getTypeId() == 2){
                const EquilateralTriangle* t = static_cast<const EquilateralTriangle*>(p.get());
                int cx = (int)std::round(t->center.x() * (width-1));
                int cy = (int)std::round(t->center.y() * (height-1));
                int rad = std::max(1, (int)std::round(t->radius * std::max(width,height)));
                cv::Scalar col(t->color.z()*255, t->color.y()*255, t->color.x()*255, t->color.w()*255);
                std::vector<cv::Point> pts;
                for (int i=0;i<3;i++) {
                    double angle = t->rotateTheta + i * 2.0 * M_PI / 3.0;
                    int px = cx + (int)std::round(rad * cos(angle));
                    int py = cy + (int)std::round(rad * sin(angle));
                    pts.push_back(cv::Point(px,py));
                }
                cv::fillConvexPoly(img, pts, col, cv::LINE_AA);
            }
            else if(p->getTypeId() == 3){
                const CurvedCapsule* cc = static_cast<const CurvedCapsule*>(p.get());
                int cx = (int)std::round(cc->center.x() * (width-1));
                int cy = (int)std::round(cc->center.y() * (height-1));
                int len = std::max(1, (int)std::round(cc->length * std::max(width,height)));
                cv::Scalar col(cc->color.z()*255, cc->color.y()*255, cc->color.x()*255, cc->color.w()*255);
                // Approximate curved capsule as a thick arc
                int radius = len/2;
                int thickness = std::max(1, (int)std::round(cc->roundParam * std::max(width,height)));
                cv::ellipse(img, cv::Point(cx,cy), cv::Size(radius,radius), cc->rotateTheta * 180.0 / M_PI, 0, 180, col, thickness, cv::LINE_AA);
            }
            else if(p->getTypeId() == 4){
                const Arc* a = static_cast<const Arc*>(p.get());
                int cx = (int)std::round(a->center.x() * (width-1));
                int cy = (int)std::round(a->center.y() * (height-1));
                int rad = std::max(1, (int)std::round(a->radius * std::max(width,height)));
                cv::Scalar col(a->color.z()*255, a->color.y()*255, a->color.x()*255, a->color.w()*255);
                double startAngle = a->rotateTheta * 180.0 / M_PI;
                double endAngle = startAngle + a->shapeTheta * 180.0 / M_PI;
                cv::ellipse(img, cv::Point(cx,cy), cv::Size(rad,rad), 0, startAngle, endAngle, col, -1, cv::LINE_AA);
            }
            else if(p->getTypeId() == 5){
                const Trapezoid* tr = static_cast<const Trapezoid*>(p.get());
                int cx = (int)std::round(tr->center.x() * (width-1));
                int cy = (int)std::round(tr->center.y() * (height-1));
                int w1 = std::max(1, (int)std::round(tr->width1 * (width-1)));
                int w2 = std::max(1, (int)std::round(tr->width2 * (width-1)));
                int h = std::max(1, (int)std::round(tr->height * (height-1)));
                cv::Scalar col(tr->color.z()*255, tr->color.y()*255, tr->color.x()*255, tr->color.w()*255);
                std::vector<cv::Point> pts;
                pts.push_back(cv::Point(cx - w1/2, cy - h/2));
                pts.push_back(cv::Point(cx + w1/2, cy - h/2));
                pts.push_back(cv::Point(cx + w2/2, cy + h/2));
                pts.push_back(cv::Point(cx - w2/2, cy + h/2));
                // Rotate points
                double angle = tr->rotateTheta;
                for (auto& pt : pts) {
                    int tx = pt.x - cx;
                    int ty = pt.y - cy;
                    int rx = (int)std::round(tx * cos(angle) - ty * sin(angle));
                    int ry = (int)std::round(tx * sin(angle) + ty * cos(angle));
                    pt.x = cx + rx;
                    pt.y = cy + ry;
                }
                cv::fillConvexPoly(img, pts, col, cv::LINE_AA);
            }
            else if(p->getTypeId() == 6){
                const Star* s = static_cast<const Star*>(p.get());
                int cx = (int)std::round(s->center.x() * (width-1));
                int cy = (int)std::round(s->center.y() * (height-1));
                int rad = std::max(1, (int)std::round(s->radius * std::max(width,height)));
                cv::Scalar col(s->color.z()*255, s->color.y()*255, s->color.x()*255, s->color.w()*255);
                std::vector<cv::Point> pts;
                int numPoints = (int)(s->theta * 2);
                for (int i=0;i<numPoints;i++) {
                    double angle = s->theta + i * M_PI / s->theta;
                    double r = (i % 2 == 0) ? rad : rad * s->externalAngle;
                    int px = cx + (int)std::round(r * cos(angle));
                    int py = cy + (int)std::round(r * sin(angle));
                    pts.push_back(cv::Point(px,py));
                }
                cv::fillConvexPoly(img, pts, col, cv::LINE_AA);
            }
            else if(p->getTypeId() == 7){
                const HalfCircle* hc = static_cast<const HalfCircle*>(p.get());
                int cx = (int)std::round(hc->center.x() * (width-1));
                int cy = (int)std::round(hc->center.y() * (height-1));
                int rad = std::max(1, (int)std::round(hc->radius * std::max(width,height)));
                cv::Scalar col(hc->color.z()*255, hc->color.y()*255, hc->color.x()*255, hc->color.w()*255);
                double startAngle = hc->rotateTheta * 180.0 / M_PI;
                double endAngle = startAngle + 180.0;
                cv::ellipse(img, cv::Point(cx,cy), cv::Size(rad,rad), 0, startAngle, endAngle, col, -1, cv::LINE_AA);
            }
            else if(p->getTypeId() == 8){
                const IsoscelesTriangle* it = static_cast<const IsoscelesTriangle*>(p.get());
                int cx = (int)std::round(it->center.x() * (width-1));
                int cy = (int)std::round(it->center.y() * (height-1));
                int w = std::max(1, (int)std::round(it->width * (width-1)));
                int h = std::max(1, (int)std::round(it->height * (height-1)));
                cv::Scalar col(it->color.z()*255, it->color.y()*255, it->color.x()*255, it->color.w()*255);
                std::vector<cv::Point> pts;
                pts.push_back(cv::Point(cx, cy - h/2));
                pts.push_back(cv::Point(cx - w/2, cy + h/2));
                pts.push_back(cv::Point(cx + w/2, cy + h/2));
                // Rotate points
                double angle = it->rotateTheta;
                for (auto& pt : pts) {
                    int tx = pt.x - cx;
                    int ty = pt.y - cy;
                    int rx = (int)std::round(tx * cos(angle) - ty * sin(angle));
                    int ry = (int)std::round(tx * sin(angle) + ty * cos(angle));
                    pt.x = cx + rx;
                    pt.y = cy + ry;
                }
                cv::fillConvexPoly(img, pts, col, cv::LINE_AA);
            }
            else if(p->getTypeId() == 9){
                const Heart* h = static_cast<const Heart*>(p.get());
                int cx = (int)std::round(h->center.x() * (width-1));
                int cy = (int)std::round(h->center.y() * (height-1));
                int size = std::max(1, (int)std::round(h->size * std::max(width,height)));
                cv::Scalar col(h->color.z()*255, h->color.y()*255, h->color.x()*255, h->color.w()*255);
                std::vector<cv::Point> pts;
                int numPoints = 100;
                for (int i=0;i<numPoints;i++) {
                    double t = (double)i / (numPoints - 1) * 2.0 * M_PI;
                    double x = 16 * sin(t)*sin(t)*sin(t);
                    double y = 13 * cos(t) - 5 * cos(2*t) - 2 * cos(3*t) - cos(4*t);
                    x = x / 16.0 * size;
                    y = -y / 16.0 * size; // Invert y for image coordinates
                    // Rotate point
                    double angle = h->rotateTheta;
                    double rx = x * cos(angle) - y * sin(angle);
                    double ry = x * sin(angle) + y * cos(angle);
                    int px = cx + (int)std::round(rx);
                    int py = cy + (int)std::round(ry);
                    pts.push_back(cv::Point(px,py));
                }
                cv::fillConvexPoly(img, pts, col, cv::LINE_AA);
            }
            else if(p->getTypeId() == 10){
                const Hexagon* he = static_cast<const Hexagon*>(p.get());
                int cx = (int)std::round(he->center.x() * (width-1));
                int cy = (int)std::round(he->center.y() * (height-1));
                int rad = std::max(1, (int)std::round(he->radius * std::max(width,height)));
                cv::Scalar col(he->color.z()*255, he->color.y()*255, he->color.x()*255, he->color.w()*255);
                std::vector<cv::Point> pts;
                for (int i=0;i<6;i++) {
                    double angle = he->rotateTheta + i * M_PI / 3.0;
                    int px = cx + (int)std::round(rad * cos(angle));
                    int py = cy + (int)std::round(rad * sin(angle));
                    pts.push_back(cv::Point(px,py));
                }
                cv::fillConvexPoly(img, pts, col, cv::LINE_AA);
            }
            else if(p->getTypeId() == 11){
                const Capsule* ca = static_cast<const Capsule*>(p.get());
                int cx = (int)std::round(ca->center.x() * (width-1));
                int cy = (int)std::round(ca->center.y() * (height-1));
                int rad = std::max(1, (int)std::round(ca->radius * std::max(width,height)));
                int len = std::max(1, (int)std::round(ca->roundParam * std::max(width,height)));
                cv::Scalar col(ca->color.z()*255, ca->color.y()*255, ca->color.x()*255, ca->color.w()*255);
                // Approximate capsule as a thick line
                cv::Point2f dir(cos(ca->rotateTheta), sin(ca->rotateTheta));
                cv::Point pt1 = cv::Point(cx,cy) - cv::Point(dir.x * len/2, dir.y * len/2);
                cv::Point pt2 = cv::Point(cx,cy) + cv::Point(dir.x * len/2, dir.y * len/2);
                cv::line(img, pt1, pt2, col, rad*2, cv::LINE_AA);

            }         
            
        }
        return img;
    }

    static MatrixX3 renderToRGB(const std::vector<std::unique_ptr<Primitive>>& prims, int width, int height, Scalar softness = 150.0) {
        Mat img = render(prims, width, height, softness);
        MatrixX3 rgb(width*height,3);
        for (int y=0;y<height;y++) for (int x=0;x<width;x++) {
            cv::Vec4b v = img.at<cv::Vec4b>(y,x);
            int idx = y*width + x;
            rgb(idx,0) = v[2]/255.0; rgb(idx,1) = v[1]/255.0; rgb(idx,2) = v[0]/255.0;
        }
        return rgb;
    }
};

// Optimizer config + stub optimizer exposing stagedOptimize used by test.cpp
struct OptimizerConfig {
    Scalar lr = 0.001;
    Scalar decayRate = 0.99;
    int decaySteps = 100;
    int maxSteps = 100;
    Scalar softness = 150.0;
    Scalar clipNorm = 1.0;
};

class PrimitiveOptimizer {
    OptimizerConfig cfg;
    std::string outDir;
public:
    PrimitiveOptimizer(OptimizerConfig c, const std::string& out) : cfg(c), outDir(out) { std::filesystem::create_directories(outDir); }
    // Convenience ctor used by test.cpp
    PrimitiveOptimizer(OptimizerConfig c) : cfg(c), outDir("output") { std::filesystem::create_directories(outDir); }
    std::vector<std::unique_ptr<Primitive>> stagedOptimize(const std::vector<std::unique_ptr<Primitive>>& initPrims, const Mat& targetImg) {
        // very small placeholder: just clone and return
        std::vector<std::unique_ptr<Primitive>> res;
        for (const auto &p : initPrims) res.push_back(p ? p->clone() : nullptr);
        return res;
    }
};


// Adam优化器
class AdamOptimizer {
private:
    OptimizerConfig config;
    std::vector<MatrixX> m, v;  // 动量缓存
    int step = 0;
    const Scalar beta1 = 0.9;
    const Scalar beta2 = 0.999;
    const Scalar eps = 1e-8;

public:
    AdamOptimizer(OptimizerConfig cfg) : config(cfg) {}

    void initMomentum(const std::vector<MatrixX>& params) {
        m.resize(params.size());
        v.resize(params.size());
        for (size_t i = 0; i < params.size(); i++) {
            m[i] = MatrixX::Zero(params[i].rows(), params[i].cols());
            v[i] = MatrixX::Zero(params[i].rows(), params[i].cols());
        }
    }

    Scalar getCurrentLR() const {
        Scalar decay = pow(config.decayRate, (Scalar)step / config.decaySteps);
        return config.lr * decay;
    }

    void clipGradients(std::vector<MatrixX>& grads) const {
        Scalar totalNorm = 0;
        for (const auto& g : grads) totalNorm += g.squaredNorm();
        totalNorm = sqrt(totalNorm);
        if (totalNorm > config.clipNorm && totalNorm > 0) {
            Scalar scale = config.clipNorm / totalNorm;
            for (auto& g : grads) g *= scale;
        }
    }

    void update(std::vector<MatrixX>& params, const std::vector<MatrixX>& grads) {
        if (m.empty()) initMomentum(params);
        Scalar lr = getCurrentLR();
        step++;

        // 梯度裁剪
        std::vector<MatrixX> clippedGrads = grads;
        clipGradients(clippedGrads);

        // Adam更新
        for (size_t i = 0; i < params.size(); i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * clippedGrads[i];
            v[i] = beta2 * v[i] + (1 - beta2) * clippedGrads[i].array().square().matrix();
            MatrixX mHat = m[i] / (1 - pow(beta1, step));
            MatrixX vHat = v[i] / (1 - pow(beta2, step));
            params[i] -= lr * mHat.array().cwiseQuotient((vHat.array().sqrt() + eps)).matrix();
        }
    }
};

// 损失函数工具
class LossUtils {
public:
    // MSE损失（RGB通道）
    static Scalar mseLoss(const MatrixX3& pred, const MatrixX3& target) {
    return (pred - target).array().square().mean();
    }

    // 边缘一致性损失（Sobel算子）
    static Scalar edgeLoss(const Mat& predImg, const Mat& targetImg) {
        Mat predGray, targetGray;
    cvtColor(predImg, predGray, cv::COLOR_RGBA2GRAY);
    cvtColor(targetImg, targetGray, cv::COLOR_RGBA2GRAY);

        Mat sobelXPred, sobelXTarget;
        Sobel(predGray, sobelXPred, CV_64F, 1, 0, 3);
        Sobel(targetGray, sobelXTarget, CV_64F, 1, 0, 3);

        Mat diff;
        absdiff(sobelXPred, sobelXTarget, diff);
    return cv::mean(diff)[0] / 255.0;
    }

    // 交点感知损失
    static Scalar intersectionLoss(
        const std::vector<std::unique_ptr<Primitive>>& prims,
        const MatrixX2& grid,
        const MatrixX3& targetRGB,
        const MatrixX3& predRGB
    ) {
        // 计算所有图元SDF
        std::vector<VectorX> sdfs;
        for (const auto& prim : prims) {
            sdfs.push_back(prim->sdf(grid));
        }

        // 排序取最近/次近SDF
        int numPixels = grid.rows();
        VectorX minSDF = VectorX::Constant(numPixels, 1e9);
        VectorX secondMinSDF = VectorX::Constant(numPixels, 1e9);
        for (const auto& sdf : sdfs) {
            for (int i = 0; i < numPixels; i++) {
                if (sdf(i) < minSDF(i)) {
                    secondMinSDF(i) = minSDF(i);
                    minSDF(i) = sdf(i);
                } else if (sdf(i) < secondMinSDF(i)) {
                    secondMinSDF(i) = sdf(i);
                }
            }
        }

        // 交点权重计算
        VectorX weight = (-minSDF * 100).array().exp().cwiseQuotient(
            (-minSDF * 100).array().exp() + 1
        ) * (-secondMinSDF * 50).array().exp().cwiseQuotient(
            (-secondMinSDF * 50).array().exp() + 1
        );

        // 加权MSE
    MatrixX3 diff = (predRGB - targetRGB).array().square().matrix();
        VectorX weightedDiff = diff.rowwise().mean().cwiseProduct(weight);
        
        // 平滑正则项
        VectorX sdfDiff = (minSDF - secondMinSDF).cwiseAbs();
        Scalar smoothReg = (-sdfDiff * 100).array().exp().mean();

        return weightedDiff.mean() + 0.1 * smoothReg;
    }
};


#endif // PRIMITIVE_RENDER_H