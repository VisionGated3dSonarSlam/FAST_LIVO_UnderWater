// Standalone tests for include/visual_frontend.hpp (OpenCV + Eigen only, no ROS).
//
//   g++ -std=c++17 -O2 -I include $(pkg-config --cflags opencv4 eigen3)
//       test/test_visual_frontend.cpp -o /tmp/test_vf $(pkg-config --libs opencv4) && /tmp/test_vf
#include <cstdio>
#include <random>

#include "visual_frontend.hpp"

using namespace visual_frontend;

static int failures = 0;
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++failures;                                                        \
            std::printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #cond);     \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                                 \
        }                                                                      \
    } while (0)

struct P { float x, y, z; };

// The original VisualFusion::build_depth_lookup, verbatim in logic, as the reference.
static cv::Mat legacy_depth(const std::vector<P> &pts, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                            const Intrinsics &k)
{
    cv::Mat d(k.height, k.width, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (const auto &p : pts)
    {
        const Eigen::Vector3d c = R * Eigen::Vector3d(p.x, p.y, p.z) + t;
        if (c.z() < 1e-3) continue;
        const int u = static_cast<int>(std::lround(k.fx * c.x() / c.z() + k.cx));
        const int v = static_cast<int>(std::lround(k.fy * c.y() / c.z() + k.cy));
        if (u < 0 || u >= k.width || v < 0 || v >= k.height) continue;
        float &s = d.at<float>(v, u);
        if (std::isnan(s) || c.z() < s) s = static_cast<float>(c.z());
    }
    return d;
}

int main()
{
    std::mt19937 rng(1);
    const Intrinsics k{572.04, 572.04, 480.0, 270.0, 960, 540};
    Eigen::Matrix3d R_cs;
    R_cs << 0.0, -1.0, 0.0, 0.258819, 0.0, -0.965926, 0.965926, 0.0, 0.258819;
    const Eigen::Vector3d t_cs(0.02148, 0.08407, -0.10618);

    // ---- projection reproduces the legacy depth map exactly
    {
        std::uniform_real_distribution<float> ux(0.5f, 8.f), uy(-4.f, 4.f), uz(-2.f, 1.f);
        std::vector<P> pts;
        for (int i = 0; i < 4000; ++i) pts.push_back({ux(rng), uy(rng), uz(rng)});
        pts.push_back({std::nanf(""), 1.f, 1.f});
        const Projection p = build_projection(pts, R_cs, t_cs, k, 3);
        const cv::Mat ref = legacy_depth(pts, R_cs, t_cs, k);
        int diff = 0, valid = 0;
        for (int v = 0; v < k.height; ++v)
            for (int u = 0; u < k.width; ++u)
            {
                const float a = p.depth.at<float>(v, u), b = ref.at<float>(v, u);
                if (std::isnan(a) != std::isnan(b) || (!std::isnan(a) && a != b)) ++diff;
                valid += !std::isnan(a);
            }
        CHECK(diff == 0, "%d pixels differ from legacy depth map", diff);
        CHECK(valid > 100, "only %d valid depth pixels", valid);
        CHECK(p.pixels.size() == p.points.size(), "pixels/points size mismatch");
        // every kept 3D point projects to its pixel
        double maxerr = 0;
        for (std::size_t i = 0; i < p.points.size(); ++i)
        {
            const auto &X = p.points[i];
            maxerr = std::max(maxerr, std::hypot(k.fx * X.x / X.z + k.cx - p.pixels[i].x,
                                                 k.fy * X.y / X.z + k.cy - p.pixels[i].y));
        }
        CHECK(maxerr < 1e-3, "pixel/point mismatch %.4f", maxerr);
        // lookup mask == where depth_at succeeds
        int mism = 0;
        for (int v = 0; v < k.height; v += 2)
            for (int u = 0; u < k.width; u += 2)
            {
                const bool m = p.lookup_mask.at<uchar>(v, u) != 0;
                const bool d = depth_at(p.depth, cv::Point2f(u, v), 3).has_value();
                mism += m != d;
            }
        CHECK(mism == 0, "lookup mask disagrees with depth_at at %d pixels", mism);
    }

    // ---- enhancement
    {
        cv::Mat img(100, 200, CV_8UC3);
        cv::randu(img, cv::Scalar(100, 120, 140), cv::Scalar(103, 122, 141));  // ~2 grey levels
        cv::Mat g0; cv::cvtColor(img, g0, cv::COLOR_BGR2GRAY);
        cv::Mat none = enhance(img, Enhance::None);
        CHECK(cv::norm(none, g0, cv::NORM_INF) == 0, "none must equal plain luminance");
        cv::Mat st = enhance(img, Enhance::Stretch);
        cv::Scalar m, sd; cv::meanStdDev(st, m, sd);
        cv::Scalar m0, sd0; cv::meanStdDev(g0, m0, sd0);
        CHECK(sd[0] > 20 * sd0[0], "stretch did not expand contrast (%.2f vs %.2f)", sd[0], sd0[0]);
        double lo, hi; cv::minMaxLoc(st, &lo, &hi);
        CHECK(lo == 0 && hi == 255, "stretch range %.0f..%.0f", lo, hi);
        CHECK(best_channel(img).data != nullptr, "best channel");
        // best channel picks the widest one: channel 0 spans 3 levels
        cv::Mat bc = best_channel(img);
        std::vector<cv::Mat> ch; cv::split(img, ch);
        CHECK(cv::norm(bc, ch[0], cv::NORM_INF) == 0, "best channel should be channel 0");
        cv::Mat sc = enhance(img, Enhance::StretchClahe);
        CHECK(sc.type() == CV_8UC1 && sc.size() == img.size(), "stretch_clahe output type");
        CHECK(parse_enhance("stretch_clahe") && !parse_enhance("bogus"), "parse");
    }

    // ---- sonar seeds + KLT on a synthetic shift
    {
        cv::Mat tex(540, 960, CV_8U);
        cv::randu(tex, 0, 255);
        cv::GaussianBlur(tex, tex, cv::Size(0, 0), 2.0);
        cv::Mat shifted;
        const cv::Mat A = (cv::Mat_<double>(2, 3) << 1, 0, 3.0, 0, 1, -2.0);  // +3 px x, -2 px y
        cv::warpAffine(tex, shifted, A, tex.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        std::vector<P> pts;
        for (float y = -2.f; y <= 2.f; y += 0.01f) pts.push_back({4.f, y, 0.0f});
        for (float y = -2.f; y <= 2.f; y += 0.01f) pts.push_back({4.f, y + 0.002f, 0.0f});  // same cells
        const Projection p = build_projection(pts, R_cs, t_cs, k, 3);
        std::vector<cv::Point2f> sp; std::vector<cv::Point3f> sx;
        select_sonar_seeds(p, tex, 3.0, 0.0, 0, sp, sx);
        CHECK(!sp.empty() && sp.size() < pts.size(), "seed dedup (%zu of %zu)", sp.size(), pts.size());
        for (std::size_t i = 0; i < sp.size(); ++i)
            for (std::size_t j = i + 1; j < std::min(sp.size(), i + 5); ++j)
                CHECK(std::floor(sp[i].x / 3) != std::floor(sp[j].x / 3) ||
                          std::floor(sp[i].y / 3) != std::floor(sp[j].y / 3),
                      "two seeds in one cell");
        std::vector<cv::Point2f> sp2; std::vector<cv::Point3f> sx2;
        select_sonar_seeds(p, tex, 3.0, 0.0, 20, sp2, sx2);
        CHECK(sp2.size() == 20, "cap: %zu", sp2.size());
        std::vector<cv::Point2f> out;
        const auto ok = track_klt(tex, shifted, sp, out, 21, 3, 1.5);
        int n = 0; double err = 0;
        for (std::size_t i = 0; i < ok.size(); ++i)
            if (ok[i]) { ++n; err = std::max(err, std::hypot(out[i].x - sp[i].x - 3.0, out[i].y - sp[i].y + 2.0)); }
        CHECK(n > static_cast<int>(sp.size()) * 8 / 10, "only %d/%zu tracked", n, sp.size());
        CHECK(err < 0.1, "KLT shift error %.3f px", err);
        // flat image: gradient filter removes every seed
        cv::Mat flat(540, 960, CV_8U, cv::Scalar(90));
        select_sonar_seeds(p, flat, 3.0, 8.0, 0, sp2, sx2);
        CHECK(sp2.empty(), "flat image should yield no seeds (%zu)", sp2.size());
    }

    // ---- gyro integration and rotation angle
    {
        GyroBuffer g(2.0);
        const Eigen::Vector3d w(0.1, -0.2, 0.3), bias(0.01, 0.0, -0.01);
        for (int i = 0; i <= 1000; ++i) g.push(i * 0.005, w);
        g.push(1.0, w);  // duplicate stamp ignored
        const auto R = g.integrate(3.5, 3.7, bias);
        CHECK(R.has_value(), "integration over a covered interval");
        if (R)
        {
            const Eigen::Matrix3d ref = so3_exp((w - bias) * 0.2);
            CHECK(rotation_angle_deg(*R * ref.transpose()) < 1e-6, "constant-rate integral");
        }
        // interval not on sample boundaries
        const auto R2 = g.integrate(3.5012, 3.6937, Eigen::Vector3d::Zero());
        CHECK(R2.has_value(), "partial interval covered");
        if (R2) CHECK(rotation_angle_deg(*R2 * so3_exp(w * (3.6937 - 3.5012)).transpose()) < 1e-6, "partial samples");
        CHECK(!g.integrate(5.0, 5.1, bias).has_value(), "outside the buffer");
        CHECK(!g.integrate(2.0, 2.1, bias).has_value(), "before the buffer (trimmed to 2 s)");

        // near-identity: the acos form gave ~0.26 deg here
        Eigen::Matrix3d C;
        C << 0.000600, -1.000000, -0.000800, 0.255700, 0.000900, -0.966700, 0.966800, 0.000400, 0.255700;
        const Eigen::Matrix3d Rb = so3_exp(Eigen::Vector3d(0.0, 0.0, 0.02));
        const Eigen::Matrix3d Rc = camera_rotation_from_body(Rb, C);
        const Eigen::Matrix3d Cq = orthonormalize(C);
        const Eigen::Matrix3d Rexact = Cq * Rb.transpose() * Cq.transpose();
        CHECK(rotation_angle_deg(Rc * Rexact.transpose()) < 1e-6, "conjugation with non-orthonormal extrinsic");
        CHECK(std::abs(rotation_angle_deg(so3_exp(Eigen::Vector3d(0, 0.01, 0))) - 0.5729578) < 1e-5, "angle value");
        CHECK(rotation_angle_deg(C * C.transpose()) < 1e-3, "near-identity non-orthonormal reads ~0 (%.4f)",
              rotation_angle_deg(C * C.transpose()));
    }

    if (failures == 0) std::printf("all visual_frontend tests passed\n");
    return failures ? 1 : 0;
}
