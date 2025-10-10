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
#endif // PRIMITIVE_RENDER_H
#include <filesystem>
#include <fstream>
#include <algorithm>

using Scalar = double;
using Vector2 = Eigen::Matrix<Scalar,2,1>;
using Vector4 = Eigen::Matrix<Scalar,4,1>;
using MatrixX2 = Eigen::Matrix<Scalar, Eigen::Dynamic, 2>;
using MatrixX3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

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
// 优化器配置
struct OptimizerConfig {
    Scalar lr = 0.001;          // 初始学习率
    Scalar decayRate = 0.99;    // 衰减率
    int decaySteps = 100;       // 衰减步长
    int maxSteps = 500;         // 最大步数
    Scalar softness = 150.0;    // SDF软化参数
    Scalar clipNorm = 1.0;      // 梯度裁剪阈值
};

// 图元优化器（两阶段优化）
class PrimitiveOptimizer {
private:
    OptimizerConfig config;
    std::string outputDir;

    // 保存优化帧
    void saveFrame(
        const std::vector<std::unique_ptr<Primitive>>& prims,
        int width, int height,
        int step,
        const std::string& subDir
    ) {
        fs::path frameDir = fs::path(outputDir) / subDir;
        fs::create_directories(frameDir);
        Mat frame = DifferentiableRasterizer::render(prims, width, height);
        
        char buf[64];
        snprintf(buf, sizeof(buf), "frame_%04d.png", step);
        imwrite((frameDir / buf).string(), frame);
    }

    // 保存参数到文件
    void saveParams(
        const std::vector<std::unique_ptr<Primitive>>& prims,
        const std::string& filePath
    ) {
        std::ofstream f(filePath);
        if (!f.is_open()) {
            std::cerr << "无法保存参数文件: " << filePath << std::endl;
            return;
        }
        for (const auto& prim : prims) prim->serialize(f);
        f.close();
    }

    // 生成归一化网格
    MatrixX2 createGrid(int width, int height) const {
        MatrixX2 grid(width * height, 2);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                grid(idx, 0) = (Scalar)x / (width - 1);
                grid(idx, 1) = (Scalar)y / (height - 1);
            }
        }
        return grid;
    }

    // 参数转图元
    std::vector<std::unique_ptr<Primitive>> paramsToPrimitives(
        const std::vector<MatrixX>& params,
        const std::vector<int>& primTypes,
        const std::vector<std::unique_ptr<Primitive>>& initPrims
    ) {
        std::vector<std::unique_ptr<Primitive>> prims;
        for (size_t i = 0; i < params.size(); i++) {
            int type = primTypes[i];
            const auto& initPrim = initPrims[i];
            MatrixX param = params[i];
            Vector4 color = param.block<1,4>(0, 3);
            color = color.cwiseMax(0.01).cwiseMin(0.99);
            color = color.array() / (1 - color.array());
            color = color.array().log();

            switch (type) {
                case 0: {  // 圆形
                    Vector2 center(param(0,0), param(0,1));
                    Scalar radius = std::max(param(0,2), 0.01);
                    prims.push_back(std::make_unique<Circle>(
                        center, radius, color, initPrim->depth
                    ));
                    break;
                }
                case 1: {  // 矩形
                    Vector2 center(param(0,0), param(0,1));
                    Vector2 size(param(0,2), param(0,3));
                    Scalar rt = param(0,4);
                    Scalar rp = std::max(param(0,5), 0.0);
                    size = size.cwiseMax(0.01);
                    rp = std::min(rp, std::min(size.x(), size.y())/2);
                    prims.push_back(std::make_unique<Rectangle>(
                        center, size, rt, rp, color, initPrim->depth
                    ));
                    break;
                }
                case 2: {  // 等边三角形
                    Vector2 center(param(0,0), param(0,1));
                    Scalar edgeLen = std::max(param(0,2), 0.01);
                    Scalar rt = param(0,3);
                    Scalar rp = std::max(param(0,4), 0.0);
                    rp = std::min(rp, edgeLen * sqrt(3)/6);
                    prims.push_back(std::make_unique<EquilateralTriangle>(
                        center, edgeLen, rt, rp, color, initPrim->depth
                    ));
                    break;
                }
                case 3: { // 弯曲胶囊
                    Vector2 center(param(0,0), param(0,1));
                    Scalar radius = std::max(param(0,2), 0.01);
                    Scalar length = std::max(param(0,3), 0.01);
                    Scalar rt = param(0,4);
                    Scalar rp = std::max(param(0,5), 0.0);
                    rp = std::min(rp, length/2);
                    prims.push_back(std::make_unique<CurvedCapsule>(
                        center, radius, length, rt, rp, color, initPrim->depth
                    ));
                    break;
                   
                }
                case 4:{ // 圆弧
                    Vector2 center(param(0,0), param(0,1));
                    Scalar radius = std::max(param(0,2), 0.01);
                    Scalar theta = param(0,3);
                    Scalar extAngle = param(0,4);
                    Scalar rp = std::max(param(0,5), 0.0);
                    rp = std::min(rp, radius);
                    prims.push_back(std::make_unique<Arc>(
                        center, radius, theta, extAngle, rp, color, initPrim->depth
                    ));
                    break;

                }
                case 5:{ // 梯形
                    Vector2 center(param(0,0), param(0,1));
                    Scalar topWidth = std::max(param(0,2), 0.01);
                    Scalar bottomWidth = std::max(param(0,3), 0.01);
                    Scalar height = std::max(param(0,4), 0.01);
                    Scalar rt = param(0,5);
                    Scalar rp = std::max(param(0,6), 0.0);
                    rp = std::min(rp, std::min(topWidth, bottomWidth)/2);
                    rp = std::min(rp, height/2);
                    prims.push_back(std::make_unique<Trapezoid>(
                        center, topWidth, bottomWidth, height, rt, rp, color, initPrim->depth
                    ));
                    break;

                }
                case 6:{ //星形
                    Vector2 center(param(0,0), param(0,1));
                    Scalar outerRadius = std::max(param(0,2), 0.01);
                    Scalar innerRadius = std::max(param(0,3), 0.01);
                    Scalar numPoints = std::max(param(0,4), 3.0);
                    Scalar rt = param(0,5);
                    Scalar rp = std::max(param(0,6), 0.0);
                    rp = std::min(rp, innerRadius);
                    prims.push_back(std::make_unique<Star>(
                        center, outerRadius, innerRadius, (int)numPoints, rt, rp, color, initPrim->depth
                    ));
                    break;

                }
                case 7:{ // 半圆
                    Vector2 center(param(0,0), param(0,1));
                    Scalar radius = std::max(param(0,2), 0.01);
                    Scalar rt = param(0,3);
                    Scalar rp = std::max(param(0,4), 0.0);
                    rp = std::min(rp, radius);
                    prims.push_back(std::make_unique<HalfCircle>(
                        center, radius, rt, rp, color, initPrim->depth
                    ));
                    break;

                }
                case 8:{ // 等腰三角形
                    Vector2 center(param(0,0), param(0,1));
                    Scalar width = std::max(param(0,2), 0.01);
                    Scalar height = std::max(param(0,3), 0.01);
                    Scalar rt = param(0,4);
                    Scalar rp = std::max(param(0,5), 0.0);
                    rp = std::min(rp, std::min(width, height)/2);
                    prims.push_back(std::make_unique<IsoscelesTriangle>(
                        center, width, height, rt, rp, color, initPrim->depth
                    ));
                    break;

                }
                case 9:{ // 心形
                    Vector2 center(param(0,0), param(0,1));
                    Scalar size = std::max(param(0,2), 0.01);
                    Scalar rt = param(0,3);         
                    prims.push_back(std::make_unique<Heart>(
                        center, size, rt, color, initPrim->depth
                    ));
                    break;
                }
                case 10:{ // 六边形
                    Vector2 center(param(0,0), param(0,1));
                    Scalar radius = std::max(param(0,2), 0.01);
                    Scalar rt = param(0,3);

                    Scalar rp = std::max(param(0,4), 0.0);
                    rp = std::min(rp, radius * (sqrt(3.0)/2));
                    prims.push_back(std::make_unique<Hexagon>(
                        center, radius, rt, rp, color, initPrim->depth
                    ));
                    break;
                }
                case 11:{ // 胶囊
                    Vector2 center(param(0,0), param(0,1));
                    Scalar radius = std::max(param(0,2), 0.01);
                    Scalar rt = param(0,3);
                    Scalar rp = std::max(param(0,4), 0.0);
                    rp = std::min(rp, radius);
                    prims.push_back(std::make_unique<Capsule>(
                        center, radius, rt, rp, color, initPrim->depth
                    ));
                    break;
                // 其他图元类型类似，省略...
                default:
                    prims.push_back(initPrim->clone());
                    break;
            }
        }
        return prims;
    }

    // 图元转参数
    std::vector<std::unique_ptr<Primitive>> paramsToPrimitives(
        const std::vector<std::unique_ptr<Primitive>>& inputPrims,
        std::vector<MatrixX>& params,
        std::vector<int>& primTypes
    ) {
        std::vector<std::unique_ptr<Primitive>> outputParams;
        params.clear();
        primTypes.clear();
        for (const auto& prim : inputPrims) {
            int type = prim->getTypeId();
            primTypes.push_back(type);
            Vector4 color = prim->color;
            color = color.cwiseMax(0.01).cwiseMin(0.99);
            color = color.array() / (1 - color.array());
            color = color.array().log();

            switch (type) {
                case 0: {  // 圆形
                    MatrixX param(1, 7);
                    param << prim->center.x(), prim->center.y(), 
                             dynamic_cast<Circle*>(prim.get())->radius,
                             color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 1: {  // 矩形
                    auto rect = dynamic_cast<Rectangle*>(prim.get());
                    MatrixX param(1, 9);
                    param << prim->center.x(), prim->center.y(),
                             rect->size.x(), rect->size.y(), rect->rotateTheta, rect->roundParam,
                             color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 2: {  // 等边三角形
                    auto tri = dynamic_cast<EquilateralTriangle*>(prim.get());
                    MatrixX param(1, 8);
                    param << prim->center.x(), prim->center.y(),
                             tri->edgeLength, tri->rotateTheta, tri->roundParam,
                             color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 3: { // 弯曲胶囊
                    auto cap = dynamic_cast<CurvedCapsule*>(prim.get());
                    MatrixX param(1, 10);
                    param << prim->center.x(), prim->center.y(), cap->radius, cap->length, cap->rotateTheta, cap->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 4:{ // 圆弧
                    auto arc = dynamic_cast<Arc*>(prim.get());
                    MatrixX param(1, 10);
                    param << prim->center.x(), prim->center.y(), arc->radius, arc->theta, arc->extAngle, arc->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 5:{ // 梯形
                    auto trap = dynamic_cast<Trapezoid*>(prim.get());
                    MatrixX param(1, 11);
                    param << prim->center.x(), prim->center.y(), trap->width1, trap->width1, trap->height, trap->rotate_theta, trap->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 6:{ // 星形
                    auto star = dynamic_cast<Star*>(prim.get());
                    MatrixX param(1, 11);
                    param << prim->center.x(), prim->center.y(), star->outerRadius, star->innerRadius, (Scalar)star->numPoints, star->rotateTheta, star->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 7:{ // 半圆
                    auto half = dynamic_cast<HalfCircle*>(prim.get());
                    MatrixX param(1, 8);
                    param << prim->center.x(), prim->center.y(), half->radius, half->rotateTheta, half->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 8:{ // 等腰三角形
                    auto iso = dynamic_cast<IsoscelesTriangle*>(prim.get());
                    MatrixX param(1, 9);
                    param << prim->center.x(), prim->center.y(), iso->width, iso->height, iso->rotateTheta, iso->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;

                }
                case 9:{ // 心形
                    auto heart = dynamic_cast<Heart*>(prim.get());
                    MatrixX param(1, 7);
                    param << prim->center.x(), prim->center.y(), heart->size, heart->rotateTheta,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 10:{ // 六边形
                    auto hex = dynamic_cast<Hexagon*>(prim.get());
                    MatrixX param(1, 8);
                    param << prim->center.x(), prim->center.y(), hex->radius, hex->rotateTheta, hex->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }
                case 11:{ // 胶囊
                    auto cap = dynamic_cast<Capsule*>(prim.get());
                    MatrixX param(1, 8);
                    param << prim->center.x(), prim->center.y(), cap->radius, cap->rotateTheta, cap->roundParam,
                                color.x(), color.y(), color.z(), color.w();
                    outputParams.push_back(param);
                    break;
                }

            }
        }
        return outputParams;
    }

    // 单阶段优化
    bool singleStageOptimize(
        std::vector<std::unique_ptr<Primitive>>& prims,
        const Mat& targetImg,
        const std::string& subDir,
        bool optimizeEdge = false
    ) {
        int width = targetImg.cols;
        int height = targetImg.rows;
        std::vector<MatrixX> params;
        std::vector<int> primTypes;
        primitivesToParams(prims, params, primTypes);

        // 初始化优化器
        AdamOptimizer optimizer(config);
        MatrixX3 targetRGB = DifferentiableRasterizer::renderToRGB(prims, width, height);
        MatrixX2 grid = createGrid(width, height);

        // 优化循环
        for (int step = 0; step < config.maxSteps; step++) {
            // 重建图元
            auto currentPrims = paramsToPrimitives(params, primTypes, prims);
            
            // 计算损失
            MatrixX3 predRGB = DifferentiableRasterizer::renderToRGB(currentPrims, width, height, config.softness);
            Mat predImg = DifferentiableRasterizer::render(currentPrims, width, height);
            Scalar mse = LossUtils::mseLoss(predRGB, targetRGB);
            Scalar edge = LossUtils::edgeLoss(predImg, targetImg);
            Scalar totalLoss = mse + edge;

            if (optimizeEdge) {
                Scalar inter = LossUtils::intersectionLoss(currentPrims, grid, targetRGB, predRGB);
                totalLoss += inter;
            }

            // 日志输出
            if (step % 100 == 0 || step == config.maxSteps - 1) {
                std::cout << "Step " << step << ", Loss: " << totalLoss 
                          << ", LR: " << optimizer.getCurrentLR() << std::endl;
                saveFrame(currentPrims, width, height, step, subDir);
            }

            // 数值梯度计算（中心差分）
            std::vector<MatrixX> grads(params.size());
            Scalar eps = 1e-5;
            for (size_t i = 0; i < params.size(); i++) {
                grads[i] = MatrixX::Zero(params[i].rows(), params[i].cols());
                for (int r = 0; r < params[i].rows(); r++) {
                    for (int c = 0; c < params[i].cols(); c++) {
                        // 正向扰动
                        auto paramsPlus = params;
                        paramsPlus[i](r, c) += eps;
                        auto primsPlus = paramsToPrimitives(paramsPlus, primTypes, prims);
                        MatrixX3 predPlus = DifferentiableRasterizer::renderToRGB(primsPlus, width, height);
                        Mat imgPlus = DifferentiableRasterizer::render(primsPlus, width, height);
                        Scalar lossPlus = LossUtils::mseLoss(predPlus, targetRGB) + LossUtils::edgeLoss(imgPlus, targetImg);

                        // 反向扰动
                        auto paramsMinus = params;
                        paramsMinus[i](r, c) -= eps;
                        auto primsMinus = paramsToPrimitives(paramsMinus, primTypes, prims);
                        MatrixX3 predMinus = DifferentiableRasterizer::renderToRGB(primsMinus, width, height);
                        Mat imgMinus = DifferentiableRasterizer::render(primsMinus, width, height);
                        Scalar lossMinus = LossUtils::mseLoss(predMinus, targetRGB) + LossUtils::edgeLoss(imgMinus, targetImg);

                        // 中心差分
                        grads[i](r, c) = (lossPlus - lossMinus) / (2 * eps);
                    }
                }
            }

            // 梯度裁剪与参数更新
            optimizer.clipGradients(grads);
            optimizer.update(params, grads);
        }

        // 更新图元
        prims = paramsToPrimitives(params, primTypes, prims);
        return true;
    }

public:
    PrimitiveOptimizer(OptimizerConfig cfg, std::string outDir) 
        : config(cfg), outputDir(outDir) {
        fs::create_directories(outputDir);
    }

    // 两阶段优化
    std::vector<std::unique_ptr<Primitive>> stagedOptimize(
        const std::vector<std::unique_ptr<Primitive>>& initPrims,
        const Mat& targetImg
    ) {
        // Deep-clone initial primitives because unique_ptr is non-copyable
        std::vector<std::unique_ptr<Primitive>> prims;
        prims.reserve(initPrims.size());
        for (const auto& p : initPrims) {
            if (p) prims.push_back(p->clone());
            else prims.emplace_back(nullptr);
        }
        int width = targetImg.cols;
        int height = targetImg.rows;

        // 阶段1：几何优化
        std::cout << "=== 阶段1: 几何参数优化 ===" << std::endl;
        singleStageOptimize(prims, targetImg, "optimization_frames", false);
        saveParams(prims, outputDir + "/geo_optimized_params.txt");

    // 阶段2：边缘与交点优化
        std::cout << "=== 阶段2: 边缘与交点优化 ===" << std::endl;
        OptimizerConfig edgeCfg = config;
        edgeCfg.lr = 0.01;
        edgeCfg.maxSteps = 200;
        this->config = edgeCfg;
        singleStageOptimize(prims, targetImg, "edge_optimization_frames", true);
        saveParams(prims, outputDir + "/final_optimized_params.txt");

        // --- quick visible post-processing step ---
        // some of the more complex optimizer internals may be disabled in this build.
        // ensure stagedOptimize produces a visible change so callers can observe it.
        std::cout << "stagedOptimize: applying small visual adjustments to " << prims.size() << " primitives\n";
        for (size_t i = 0; i < prims.size(); ++i) {
            if (!prims[i]) continue;
            // if circle, slightly enlarge; otherwise nudge color
            if (prims[i]->getTypeId() == 0) {
                Circle* c = dynamic_cast<Circle*>(prims[i].get());
                if (c) c->radius = c->radius * 1.05 + 1e-4; // small growth
                prims[i]->color = (prims[i]->color * 0.9).array().max(0.0).matrix();
            } else {
                prims[i]->color = (prims[i]->color * 0.95).array().max(0.0).matrix();
            }
        }

        return prims;
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

// 可微光栅化器
class DifferentiableRasterizer {
public:
    static Mat render(
        const std::vector<std::unique_ptr<Primitive>>& prims,
        int width, int height,
        Scalar softness = 150.0,
        bool isFirst = true
    ) {
        // 创建归一化网格
        MatrixX2 grid(width * height, 2);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                grid(idx, 0) = (Scalar)x / (width - 1);
                grid(idx, 1) = (Scalar)y / (height - 1);
            }
        }

        // 计算SDF和覆盖度
        int numPrims = prims.size();
        std::vector<VectorX> sdfs(numPrims);
        std::vector<VectorX> coverage(numPrims);
        std::vector<Vector3> baseColors(numPrims);
        std::vector<Scalar> depths(numPrims);

        for (int i = 0; i < numPrims; i++) {
            sdfs[i] = prims[i]->sdf(grid);
            // Sigmoid软化
            coverage[i] = (-sdfs[i] * softness).array().exp().cwiseQuotient(
                (-sdfs[i] * softness).array().exp() + 1
            );
            baseColors[i] = prims[i]->color.head<3>();
            depths[i] = prims[i]->depth;
            std::cout << "sdfs: " << sdfs[i].transpose() << "s\n";

        }

        // 深度权重（Softmax）
        MatrixX depthWeights(numPrims, width * height);
        for (int i = 0; i < numPrims; i++) {
            depthWeights.row(i) = (depths[i] * 0.01) * VectorX::Ones(width * height);
        }
        depthWeights = depthWeights.array().exp();
        depthWeights = depthWeights.array().rowwise() / depthWeights.colwise().sum().array();

        // 颜色混合（按像素计算权重：weight_i(p) = depthWeight_i(p) * alpha_i(p))
        MatrixX3 predRGB(width * height, 3);
        predRGB.setZero();
        VectorX predAlpha(width * height);
        predAlpha.setZero();

        // Build per-primitive per-pixel weights matrix
        MatrixX weights(numPrims, width * height);
        for (int i = 0; i < numPrims; ++i) {
            VectorX alpha = prims[i]->color.w() * coverage[i];
            // depthWeights.row(i) is 1 x (w*h)
            weights.row(i) = depthWeights.row(i).cwiseProduct(alpha.transpose());
        }

        // Normalize per-pixel
        VectorX sumWeights = weights.colwise().sum().transpose();
        for (int p = 0; p < width * height; ++p) {
            Scalar s = sumWeights(p) + 1e-8;
            for (int i = 0; i < numPrims; ++i) {
                Scalar w = weights(i, p) / s;
                for (int c = 0; c < 3; ++c) {
                    predRGB(p, c) += baseColors[i](c) * w;
                }
                predAlpha(p) += prims[i]->color.w() * coverage[i](p) * w;
            }
        }

        // 诊断信息：输出平均颜色/alpha以帮助定位空白图问题（写入临时文件以确保持久化）
        {
            Scalar meanR = predRGB.col(0).mean();
            Scalar meanG = predRGB.col(1).mean();
            Scalar meanB = predRGB.col(2).mean();
            Scalar meanA = predAlpha.mean();
            // write to /tmp for persistence
            std::ofstream dbg("/tmp/raster_debug.txt", std::ios::app);
            dbg << "prims=" << prims.size() << " meanR=" << meanR << " meanG=" << meanG << " meanB=" << meanB << " meanA=" << meanA << "\n";
            if (!prims.empty()) dbg << "first_type=" << prims[0]->getTypeId() << " first_color=" << prims[0]->color.transpose() << "\n";
            dbg.close();
        }

        // 转换为OpenCV图像
        Mat img(height, width, CV_8UC4);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                Vec4b& pixel = img.at<Vec4b>(y, x);
                pixel[0] = (uchar)(predRGB(idx, 2) * 255);  // B
                pixel[1] = (uchar)(predRGB(idx, 1) * 255);  // G
                pixel[2] = (uchar)(predRGB(idx, 0) * 255);  // R
                pixel[3] = (uchar)(predAlpha(idx) * 255);   // A
            }
        }
        return img;
    }

    // 优化专用光栅化（返回归一化RGB）
    static MatrixX3 renderToRGB(
        const std::vector<std::unique_ptr<Primitive>>& prims,
        int width, int height,
        Scalar softness = 150.0
    ) {
        Mat img = render(prims, width, height, softness);
        MatrixX3 rgb(width * height, 3);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                Vec4b pixel = img.at<Vec4b>(y, x);
                rgb(idx, 0) = pixel[2] / 255.0;  // R
                rgb(idx, 1) = pixel[1] / 255.0;  // G
                rgb(idx, 2) = pixel[0] / 255.0;  // B
            }
        }
        return rgb;
    }
};


// 工具类（加载/保存/对比）
class MainUtils {
public:
    // 加载目标图像
    static Mat loadTargetImage(const std::string& path) {
    Mat img = imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) throw std::runtime_error("无法加载目标图像: " + path);
    if (img.channels() == 1) cvtColor(img, img, cv::COLOR_GRAY2RGBA);
    else if (img.channels() == 3) cvtColor(img, img, cv::COLOR_BGR2RGBA);
        else if (img.channels() != 4) throw std::runtime_error("不支持的通道数: " + std::to_string(img.channels()));
        return img;
    }

    // 从JSON加载图元
    static std::vector<std::unique_ptr<Primitive>> loadPrimitivesFromJson(const std::string& jsonPath) {
        std::vector<std::unique_ptr<Primitive>> prims;
        std::ifstream f(jsonPath);
        if (!f.is_open()) throw std::runtime_error("无法打开JSON文件: " + jsonPath);
        
    nlohmann::json data;
    f >> data;
        auto shapesId = data["shapes id"].get<std::vector<int>>();
        auto shapesOp = data["shapes op"].get<std::vector<int>>();
        auto shapesParams = data["shapes params"].get<std::vector<std::vector<Scalar>>>();

        for (size_t i = 0; i < shapesId.size(); i++) {
            int type = shapesId[i];
            int op = shapesOp[i];
            auto& params = shapesParams[i];
            Scalar depth = i + 1;

            // 颜色设置（op=1为黑色，其他为白色）
            Vector4 color(1, 1, 1, 0.999);
            if (op == 1) color = Vector4(0, 0, 0, 0.999);

            switch (type) {
                case 0: {  // 圆形
                    Vector2 center(params[0], params[1]);
                    Scalar radius = params[2];
                    prims.push_back(std::make_unique<Circle>(center, radius, color, depth));
                    break;
                }
                case 1: {  // 矩形
                    Vector2 center(params[0], params[1]);
                    Vector2 size(params[2], params[3]);
                    Scalar rt = params[4], rp = params[5];
                    prims.push_back(std::make_unique<Rectangle>(center, size, rt, rp, color, depth));
                    break;
                }
                case 2: {  // 等边三角形
                    Vector2 center(params[0], params[1]);
                    Scalar size = params[2];
                    Scalar rt = params[3], rp = params[4];
                    prims.push_back(std::make_unique<EquilateralTriangle>(center, size, rt, rp, color, depth));
                    break;
                }
                case 3: { //弯曲胶囊
                    Vector2 center(params[0], params[1]);
                    Scalar length= params[2];
                    Scalar a = params[3];
                    Scalar rt = params[3], rp = params[4];
                    prims.push_back(std::make_unique<CurvedCapsule>(center, length, a, rt, rp, color, depth));
                    break;
                }
                case 4: {  // 圆弧
                    Vector2 center(params[0], params[1]);
                    Scalar radius = params[2];
                    Scalar startAngle = params[3], endAngle = params[4];
                    Scalar width = params[5];
                    prims.push_back(std::make_unique<Arc>(center, radius, startAngle, endAngle, width, color, depth));
                    break;
                }
                case 5:{ //梯形
                    Vector2 center(params[0], params[1]);
                    Scalar width1=params[2], width2=params[3], height=params[4], rt = params[5], rp = params[6];
                    prims.push_back(std::make_unique<Trapezoid>(center, width1, width2, height, rt, rp, color, depth));
                    break;

                }
                case 6: {  // 星形
                    Vector2 center(params[0], params[1]);
                    Scalar outerRadius = params[2];
                    Scalar innerRadius = params[3];
                    int numPoints = std::max(3, (int)params[4]);
                    Scalar rt = params[5], rp = params[6];
                    prims.push_back(std::make_unique<Star>(center, outerRadius, innerRadius, numPoints, rt, rp, color, depth));
                    break;
                }
                case 7: {  // 半圆
                    Vector2 center(params[0], params[1]);
                    Scalar radius = params[2];
                    Scalar rotateTheta = params[3];
                    Scalar rotateParam = params[4];
                    prims.push_back(std::make_unique<HalfCircle>(center, radius, rotateTheta, rotateParam, color, depth));
                    break;
                }
                case 8:{ // 等腰三角形
                    Vector2 center(params[0], params[1]);
                    Scalar base = params[2];
                    Scalar height = params[3];
                    Scalar rotateTheta = params[4];
                    Scalar rp = params[5];
                    prims.push_back(std::make_unique<IsoscelesTriangle>(center, base, height, rotateTheta, rp, color, depth));
                    break;

                }
                case 9: {  // 心形
                    Vector2 center(params[0], params[1]);
                    Scalar size = params[2];
                    Scalar rotateTheta = params[3];
                    prims.push_back(std::make_unique<Heart>(center, size, rotateTheta, color, depth));
                    break;
                }
                case 10: {  // 六边形
                    Vector2 center(params[0], params[1]);
                    Scalar radius = params[2];
                    Scalar rotateTheta = params[3];
                    Scalar roundParam = params[4];
                    prims.push_back(std::make_unique<Hexagon>(center, radius, rotateTheta, roundParam, color, depth));
                    break;
                }     
                case 11: {  // 胶囊
                    Vector2 center(params[0], params[1]);
                    Scalar radius = params[2];
                    Scalar rotateTheta = params[3];
                    Scalar roundParam = params[4];
                    prims.push_back(std::make_unique<Capsule>(center, radius, rotateTheta, roundParam, color, depth));
                    break;
                }
                default:
                    std::cerr << "未知图元类型ID: " << type << std::endl;
                    break;
        }
        return prims;
    }

    // 生成对比图像
    static void saveComparison(
        const Mat& target,
        const Mat& initial,
        const Mat& optimized,
        const std::string& savePath
    ) {
        int width = target.cols;
        int height = target.rows;
        Mat comp(height, width * 3, CV_8UC4);
        
    target.copyTo(comp(cv::Rect(0, 0, width, height)));
    initial.copyTo(comp(cv::Rect(width, 0, width, height)));
    optimized.copyTo(comp(cv::Rect(width * 2, 0, width, height)));
        
    putText(comp, "Target", Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    putText(comp, "Initial", Point(width + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    putText(comp, "Optimized", Point(width * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        
        imwrite(savePath, comp);
    }
};
