#ifndef VISUAL_FRONTEND_HPP
#define VISUAL_FRONTEND_HPP

// ROS-free building blocks for the visual front end (OpenCV + Eigen only), split out of
// visual_fusion.hpp so they can be unit-tested and checked against the offline bench
// (thesis_vio/scripts/frontend_bench.py) without a ROS build.
//
// What lives here, and why each piece exists (numbers from the 2026-10-01 bench, t6 semi):
//
//   enhance()         Semi-turbid frames span ~2 grey levels (std 1.8), yet the scene is
//                     still there. A percentile stretch of the highest-contrast channel plus
//                     CLAHE recovers it: good measurements 8.5% -> 73.6%.
//   build_projection  Sonar cloud -> sparse depth image (same rules as the original
//                     build_depth_lookup), plus the projected pixel/3D point of every return
//                     and a mask of where a +-3 px depth lookup can succeed.
//   select_sonar_seeds / track_klt
//                     Track the projected sonar returns themselves: every correspondence
//                     sits on real structure and carries an exact depth. ~100% good in clear
//                     water at the cost of the current front end.
//   GyroBuffer / rotation_angle_deg
//                     The PnP rotation vs integrated-gyro rotation disagreement predicts a
//                     bad velocity measurement far better than reprojection error
//                     (Spearman 0.73-0.97 vs ~0.5), and needs no filter state.

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace visual_frontend
{

// ----------------------------------------------------------------------------- enhancement

enum class Enhance
{
    None,          ///< luminance only, exactly the original behaviour
    Clahe,         ///< highest-contrast channel + CLAHE(clip)
    Stretch,       ///< highest-contrast channel, 0.5-99.5 percentile stretch to 0..255
    StretchClahe,  ///< stretch, then CLAHE(clip)
};

inline std::optional<Enhance> parse_enhance(const std::string &s)
{
    if (s == "none") return Enhance::None;
    if (s == "clahe") return Enhance::Clahe;
    if (s == "stretch") return Enhance::Stretch;
    if (s == "stretch_clahe") return Enhance::StretchClahe;
    return std::nullopt;
}

/// Channel with the largest standard deviation. Underwater, red attenuates first, so the
/// luminance mix wastes contrast; picking the best channel per frame keeps what is left.
inline cv::Mat best_channel(const cv::Mat &img)
{
    if (img.channels() == 1)
    {
        return img;
    }
    std::vector<cv::Mat> ch;
    cv::split(img, ch);
    int best = 0;
    double best_std = -1.0;
    for (int i = 0; i < std::min<int>(3, static_cast<int>(ch.size())); ++i)
    {
        cv::Scalar mean, stddev;
        cv::meanStdDev(ch[static_cast<std::size_t>(i)], mean, stddev);
        if (stddev[0] > best_std)
        {
            best_std = stddev[0];
            best = i;
        }
    }
    return ch[static_cast<std::size_t>(best)];
}

/// Linear stretch so the [lo_pct, hi_pct] percentiles map to [0, 255]. 8-bit input.
inline cv::Mat percentile_stretch(const cv::Mat &gray, double lo_pct = 0.5, double hi_pct = 99.5)
{
    CV_Assert(gray.type() == CV_8UC1);
    int hist[256] = {0};
    for (int r = 0; r < gray.rows; ++r)
    {
        const uchar *p = gray.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c) ++hist[p[c]];
    }
    const double n = static_cast<double>(gray.total());
    auto percentile = [&](double pct) {
        const double target = pct / 100.0 * (n - 1);
        double acc = 0.0;
        for (int v = 0; v < 256; ++v)
        {
            acc += hist[v];
            if (acc > target) return static_cast<double>(v);
        }
        return 255.0;
    };
    const double lo = percentile(lo_pct);
    const double hi = percentile(hi_pct);
    const double gain = 255.0 / std::max(hi - lo, 1.0);
    cv::Mat out;
    gray.convertTo(out, CV_8U, gain, -lo * gain);  // saturate_cast clips to 0..255
    return out;
}

/// Image used for TRACKING. `img` is the already-downscaled decoded frame (1, 3 or 4
/// channels). Image-quality metrics for the gate are computed elsewhere on the raw
/// luminance, so enhancement never changes what the visibility check measures.
inline cv::Mat enhance(const cv::Mat &img, Enhance mode, double clahe_clip = 2.0)
{
    if (mode == Enhance::None)
    {
        cv::Mat g;
        if (img.channels() == 3) cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4) cv::cvtColor(img, g, cv::COLOR_BGRA2GRAY);
        else g = img.clone();
        return g;
    }
    cv::Mat c = best_channel(img);
    if (mode == Enhance::Stretch || mode == Enhance::StretchClahe)
    {
        c = percentile_stretch(c);
    }
    if (mode == Enhance::Clahe || mode == Enhance::StretchClahe)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clahe_clip, cv::Size(8, 8));
        cv::Mat o;
        clahe->apply(c, o);
        return o;
    }
    return c.clone();
}

// ----------------------------------------------------------------------------- sonar projection

struct Intrinsics
{
    double fx, fy, cx, cy;
    int width, height;
};

/// Sonar returns projected into one (downscaled) image.
struct Projection
{
    cv::Mat depth;                    ///< CV_32F, NaN where no return projects (nearest wins)
    std::vector<cv::Point2f> pixels;  ///< sub-pixel projection of each kept return
    std::vector<cv::Point3f> points;  ///< same returns, camera frame, metres
    cv::Mat lookup_mask;              ///< CV_8U, 255 where depth_at(+-radius) can succeed

    bool empty() const { return depth.empty(); }
};

/// Same projection rules as the original VisualFusion::build_depth_lookup (z > 1e-3,
/// rounded pixel, nearest return per pixel), so the legacy path's depth map is unchanged.
template <typename PointRange>
inline Projection build_projection(const PointRange &sonar_points, const Eigen::Matrix3d &R_cam_sonar,
                                   const Eigen::Vector3d &t_cam_sonar, const Intrinsics &k,
                                   int lookup_radius_px = 3)
{
    Projection p;
    if (k.width <= 0 || k.height <= 0) return p;
    p.depth = cv::Mat(k.height, k.width, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (const auto &sp : sonar_points)
    {
        const Eigen::Vector3d s(sp.x, sp.y, sp.z);
        const Eigen::Vector3d c = R_cam_sonar * s + t_cam_sonar;
        if (!(c.z() >= 1e-3)) continue;  // also drops NaN
        const double uf = k.fx * c.x() / c.z() + k.cx;
        const double vf = k.fy * c.y() / c.z() + k.cy;
        const long u = std::lround(uf);
        const long v = std::lround(vf);
        if (u < 0 || u >= k.width || v < 0 || v >= k.height) continue;
        float &stored = p.depth.at<float>(static_cast<int>(v), static_cast<int>(u));
        const float z = static_cast<float>(c.z());
        if (std::isnan(stored) || z < stored) stored = z;
        p.pixels.emplace_back(static_cast<float>(uf), static_cast<float>(vf));
        p.points.emplace_back(static_cast<float>(c.x()), static_cast<float>(c.y()), z);
    }
    cv::Mat valid;
    cv::compare(p.depth, p.depth, valid, cv::CMP_EQ);  // NaN != NaN
    const int r = std::max(0, lookup_radius_px);
    if (r > 0)
    {
        cv::dilate(valid, p.lookup_mask,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * r + 1, 2 * r + 1)));
    }
    else
    {
        p.lookup_mask = valid;
    }
    return p;
}

/// Nearest valid depth within +-radius px (the original depth_at()).
inline std::optional<double> depth_at(const cv::Mat &depth, const cv::Point2f &pixel, int radius = 3)
{
    if (depth.empty()) return std::nullopt;
    const int u = static_cast<int>(std::lround(pixel.x));
    const int v = static_cast<int>(std::lround(pixel.y));
    double best_depth = 0.0;
    double best_distance = std::numeric_limits<double>::max();
    for (int dv = -radius; dv <= radius; ++dv)
    {
        const int row = v + dv;
        if (row < 0 || row >= depth.rows) continue;
        for (int du = -radius; du <= radius; ++du)
        {
            const int col = u + du;
            if (col < 0 || col >= depth.cols) continue;
            const float value = depth.at<float>(row, col);
            if (std::isnan(value)) continue;
            const double d = std::sqrt(static_cast<double>(du * du + dv * dv));
            if (d < best_distance)
            {
                best_distance = d;
                best_depth = value;
            }
        }
    }
    if (best_distance == std::numeric_limits<double>::max()) return std::nullopt;
    return best_depth;
}

// ----------------------------------------------------------------------------- sonar seeds

/// One seed per occupied cell of a `spacing_px` grid (nearest return kept), optionally only
/// where the tracking image has gradient (a flat patch cannot be tracked), capped at
/// `max_seeds` by a deterministic stride.
inline void select_sonar_seeds(const Projection &proj, const cv::Mat &gray, double spacing_px,
                               double min_gradient, int max_seeds,
                               std::vector<cv::Point2f> &seed_px, std::vector<cv::Point3f> &seed_xyz)
{
    seed_px.clear();
    seed_xyz.clear();
    if (proj.pixels.empty()) return;
    cv::Mat mag;
    if (min_gradient > 0.0 && !gray.empty())
    {
        cv::Mat gx, gy;
        cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
        cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
    }
    const double s = std::max(1.0, spacing_px);
    struct Cand { long key; float z; std::size_t idx; };
    std::vector<Cand> cands;
    cands.reserve(proj.pixels.size());
    for (std::size_t i = 0; i < proj.pixels.size(); ++i)
    {
        const cv::Point2f &px = proj.pixels[i];
        if (!mag.empty())
        {
            const int u = std::clamp(static_cast<int>(std::lround(px.x)), 0, mag.cols - 1);
            const int v = std::clamp(static_cast<int>(std::lround(px.y)), 0, mag.rows - 1);
            if (mag.at<float>(v, u) < min_gradient) continue;
        }
        const long cu = static_cast<long>(std::floor(px.x / s));
        const long cv_ = static_cast<long>(std::floor(px.y / s));
        cands.push_back({cv_ * 100000L + cu, proj.points[i].z, i});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
        return a.key != b.key ? a.key < b.key : a.z < b.z;
    });
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < cands.size(); ++i)
    {
        if (i == 0 || cands[i].key != cands[i - 1].key) keep.push_back(cands[i].idx);
    }
    const std::size_t cap = max_seeds > 0 ? static_cast<std::size_t>(max_seeds) : keep.size();
    const double stride = keep.size() > cap ? static_cast<double>(keep.size()) / cap : 1.0;
    for (double f = 0.0; f < static_cast<double>(keep.size()) && seed_px.size() < cap; f += stride)
    {
        const std::size_t idx = keep[static_cast<std::size_t>(f)];
        seed_px.push_back(proj.pixels[idx]);
        seed_xyz.push_back(proj.points[idx]);
    }
}

/// Pyramidal KLT with a forward-backward check. Returns per-point success and fills `out`.
inline std::vector<uchar> track_klt(const cv::Mat &prev, const cv::Mat &curr,
                                    const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &out,
                                    int window, int levels, double max_fb_px)
{
    std::vector<uchar> ok(pts.size(), 0);
    out.clear();
    if (pts.empty()) return ok;
    std::vector<uchar> st1, st2;
    std::vector<float> e1, e2;
    std::vector<cv::Point2f> back;
    const cv::Size win(window, window);
    cv::calcOpticalFlowPyrLK(prev, curr, pts, out, st1, e1, win, levels);
    cv::calcOpticalFlowPyrLK(curr, prev, out, back, st2, e2, win, levels);
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        ok[i] = st1[i] && st2[i] && cv::norm(pts[i] - back[i]) <= max_fb_px;
    }
    return ok;
}

// ----------------------------------------------------------------------------- gyro

inline Eigen::Matrix3d so3_exp(const Eigen::Vector3d &w)
{
    const double th = w.norm();
    if (th < 1e-12) return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}

/// Nearest rotation matrix (via a normalised quaternion). Calibrated extrinsics read from
/// YAML are rounded and only orthonormal to ~1e-4; that is harmless for projection but not
/// for comparing two nearly identical rotations.
inline Eigen::Matrix3d orthonormalize(const Eigen::Matrix3d &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d Q = svd.matrixU() * svd.matrixV().transpose();
    if (Q.determinant() < 0.0)
    {
        Eigen::Matrix3d U = svd.matrixU();
        U.col(2) *= -1.0;
        Q = U * svd.matrixV().transpose();
    }
    return Q;
}

/// Rotation angle in degrees, well conditioned near zero. The textbook acos((tr R - 1)/2)
/// is not: a 1e-5 non-orthonormality reads as ~0.26 deg, which is the size of the gate
/// threshold. (Found exactly that floor on the calibrated sim extrinsic.)
inline double rotation_angle_deg(const Eigen::Matrix3d &R)
{
    Eigen::Quaterniond q(orthonormalize(R));
    q.normalize();
    return 2.0 * std::atan2(q.vec().norm(), std::abs(q.w())) * 180.0 / M_PI;
}

/// Short history of body-frame gyro samples, so the rotation between two IMAGE stamps can
/// be integrated even though FAST-LIO only keeps the current scan's IMU messages.
class GyroBuffer
{
public:
    explicit GyroBuffer(double keep_seconds = 2.0) : keep_(keep_seconds) {}

    void push(double t, const Eigen::Vector3d &w)
    {
        if (!samples_.empty() && t <= samples_.back().t) return;  // duplicates / reordering
        samples_.push_back({t, w});
        while (!samples_.empty() && samples_.front().t < t - keep_) samples_.pop_front();
    }

    /// R_b0_b1: orientation of the body at t1 relative to t0 (R_wb(t1) = R_wb(t0) * R).
    /// Zero-order hold on each sample, clipped to [t0, t1]. nullopt when the buffer does not
    /// cover the interval to within `max_gap` seconds at either end.
    std::optional<Eigen::Matrix3d> integrate(double t0, double t1, const Eigen::Vector3d &bias,
                                             double max_gap = 0.02) const
    {
        if (samples_.size() < 2 || t1 <= t0) return std::nullopt;
        if (samples_.front().t > t0 + max_gap || samples_.back().t < t1 - max_gap) return std::nullopt;
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        for (std::size_t k = 0; k + 1 < samples_.size(); ++k)
        {
            const double a = std::max(samples_[k].t, t0);
            const double b = std::min(samples_[k + 1].t, t1);
            if (b <= a) continue;
            R = R * so3_exp((samples_[k].w - bias) * (b - a));
        }
        // tail beyond the last sample (within max_gap): hold the last rate
        if (samples_.back().t < t1)
        {
            R = R * so3_exp((samples_.back().w - bias) * (t1 - samples_.back().t));
        }
        return R;
    }

    std::size_t size() const { return samples_.size(); }

private:
    struct Sample { double t; Eigen::Vector3d w; };
    std::deque<Sample> samples_;
    double keep_;
};

/// Camera POINT transform (X_curr = R X_prev) implied by a body rotation R_b0_b1.
inline Eigen::Matrix3d camera_rotation_from_body(const Eigen::Matrix3d &R_b0_b1,
                                                 const Eigen::Matrix3d &R_cam_body)
{
    const Eigen::Matrix3d C = orthonormalize(R_cam_body);
    return C * R_b0_b1.transpose() * C.transpose();
}

}  // namespace visual_frontend

#endif  // VISUAL_FRONTEND_HPP
