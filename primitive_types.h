// primitive_types.h
#ifndef PRIMITIVE_TYPES_H
#define PRIMITIVE_TYPES_H
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <chrono>
#include <nlohmann/json.hpp> 
#include <vector>
#include <array>
#include <cmath>
#include <memory>

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
struct Vec2 {
    double x, y;
    Vec2(double x = 0, double y = 0) : x(x), y(y) {}
    
    Vec2 operator+(const Vec2& other) const { return Vec2(x + other.x, y + other.y); }
    Vec2 operator-(const Vec2& other) const { return Vec2(x - other.x, y - other.y); }
    Vec2 operator*(double scalar) const { return Vec2(x * scalar, y * scalar); }
    double dot(const Vec2& other) const { return x * other.x + y * other.y; }
    double length() const { return std::sqrt(x*x + y*y); }
    Vec2 normalize() const { double len = length(); return len > 0 ? Vec2(x/len, y/len) : Vec2(0,0); }
};

struct Color {
    double r, g, b, a;
    Color(double r = 0, double g = 0, double b = 0, double a = 1) : r(r), g(g), b(b), a(a) {}
};

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

    // Legacy-compatible constructor: accept old Vec2 and Color types
    Circle(const ::Vec2& c, double r, const ::Color& col, double d)
        : Primitive(Vector2(c.x, c.y), Vector4(col.r, col.g, col.b, col.a), d), radius(r) {}

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
    int getTypeId() const override { return 12; }  // 假设12是弯曲胶囊体的唯一ID

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
    int getTypeId() const override { return 13; }

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

    #endif // PRIMITIVE_TYPES_H