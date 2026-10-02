#ifndef VISUAL_FUSION_HPP
#define VISUAL_FUSION_HPP

// Monocular visual pose refinement for the sonar-inertial ESIKF.
//
// Design notes
// ------------
// The measurement is a *relative* body motion between two camera frames, converted into a
// position residual and injected through the same linear-update path the DVL and pressure
// sensors already use (auxiliary_sensor_fusion.hpp). Reusing that path means the visual
// update inherits its Joseph-form covariance handling and innovation gating rather than
// re-deriving them.
//
// Feature-based, not direct photometric. Direct methods align raw intensity patches and so
// depend on image gradients, which the turbidity validation showed collapse ~99.97% between
// clear and murky water; they also assume denser, cleaner geometry than a 3D sonar cloud
// provides. Sparse features with a fundamental-matrix / PnP front end degrade more
// gracefully and fail in a way that is easy to detect and report.
//
// Depth comes from the sonar cloud, projected into the image using the calibrated
// camera<-sonar extrinsic. That makes this genuinely sonar-visual rather than monocular:
// the scale is observed, not estimated, so there is no scale drift to chase.
//
// This class computes and applies the update but deliberately makes no decision about
// *whether* vision should be trusted -- that is the visibility gate's job (Phase 4). Here
// the update is unconditional, which is exactly the "always-on VIO" arm of the evaluation.

#include <deque>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "common_lib.h"
#include "use-ikfom.hpp"
#include "visual_frontend.hpp"

class VisualFusion
{
public:
    // Track the process-noise dimension symbolically rather than hardcoding it: upstream
    // uses process_noise_ikfom::DOF (12), the earlier fork used a literal 15, and a mismatch
    // here is a compile error at best and a silently wrong filter at worst.
    using Ekf = esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom>;

    struct Params
    {
        bool enabled = false;

        // Camera intrinsics (pinhole). Defaults match the OceanSim underwater camera in
        // MonoRgbCam_ThreeDSonar_Extrinsic_calibration/config/calibration_params.json.
        double fx = 1144.0834;
        double fy = 1144.0834;
        double cx = 960.0;
        double cy = 540.0;
        int image_width = 1920;
        int image_height = 1080;

        // camera <- sonar extrinsic, from the calibration package.
        M3D R_cam_sonar = M3D::Identity();
        V3D t_cam_sonar = V3D::Zero();

        int max_features = 300;
        double feature_quality = 0.01;
        double min_feature_distance = 12.0;
        int klt_window = 21;
        int klt_pyramid_levels = 3;

        // Reject a tracked feature whose forward-backward reprojection disagrees by more
        // than this many pixels. Cheap and very effective against the spurious matches
        // that backscatter produces.
        double max_fb_error_px = 1.5;

        // Minimum tracked features for the update to be attempted at all.
        int min_tracked_features = 30;
        // Minimum features with valid sonar depth for a metric translation estimate.
        int min_depth_features = 12;

        double max_reprojection_error_px = 3.0;
        // Reject an update implying motion faster than this. A gross visual failure shows
        // up as an implausible jump long before it shows up as a large residual.
        double max_translation_per_frame_m = 1.0;

        double position_cov = 0.05;      // metres^2
        double innovation_gate_sigma = 3.0;

        // Which gate decides whether a measurement is used.
        //   "quality"    : per-measurement quality + a state-independent speed bound (default)
        //   "innovation" : legacy 3-sigma test on the residual, threshold derived from
        //                  position_cov -- kept only for A/B comparison
        //   "none"       : accept everything that survives PnP (calibration runs)
        //
        // The innovation gate ties two jobs to one parameter: position_cov sets both the
        // update weight and the rejection threshold, so trusting vision more also narrows the
        // gate and throws vision away. It also cannot tell "this measurement is bad" from "my
        // state is bad" -- the residual is large either way -- so in a diverging run it rejects
        // good measurements exactly when they are needed. Measured on test6: 64% of
        // measurements rejected at position_cov 3e-5, and rejections RISING again at 3e-4
        // despite a 3x wider gate. The quality gate reads only properties of the measurement
        // itself, so it cannot punish the filter for being lost.
        std::string gate_mode = "quality";
        //
        // Defaults calibrated against ground truth (test2 + test6, gate off, 666 measurements).
        // Clear-water measurements are already good -- median error 0.05-0.07 m/s, only 5% worse
        // than 0.3 m/s -- so the within-stream checks are deliberately light: inliers < 20 would
        // catch 16 of 24 bad measurements but discard 32% of good ones, which is the same trade
        // the innovation gate made. Turbidity is where the gate earns its keep: semi-turbid
        // measurements have median error 0.74 m/s and 46 deg direction error, and image quality
        // separates them from clear by ~20x (sharpness 2.0 vs >=39 at p5; contrast 4.7 vs >=19.5).
        int gate_min_inliers = 15;           // PnP RANSAC inliers
        double gate_min_inlier_ratio = 0.0;  // inliers / depth features; 0 disables (uninformative)
        double gate_max_reproj_px = 1.0;     // mean inlier reprojection error, downscaled px
        double gate_min_sharpness = 10.0;    // Laplacian variance
        double gate_min_contrast = 10.0;     // grey-level std-dev
        double gate_max_speed = 2.0;         // measured camera speed, m/s; 0 disables
        // Optional loose bound on |residual|. State-DEPENDENT, so off by default: it is the
        // one check that can reintroduce the failure the quality gate exists to remove.
        double gate_max_residual = 0.0;

        // Per-measurement CSV for calibrating the thresholds against ground truth.
        std::string debug_csv;

        // Downscale before tracking. The 1920x1080 underwater camera is far higher
        // resolution than feature tracking needs, and the sonar cloud is sparse enough
        // that sub-pixel precision is not the limiting factor.
        double image_scale = 0.5;

        double max_image_dt = 0.25;      // seconds between usable frames
        // Reject an image older than this relative to the scan it would be paired with (s).
        // The newest image at or before the scan is used, which on its own accepts ANY age:
        // on 2026-10-02 the semi-turbid stream replayed from a second bag lagged ~2.8 s, so
        // every semi measurement paired an image with sonar depth from a different moment
        // (65% bad updates) and nothing flagged it. One scan period is a safe limit. 0 disables.
        double max_image_age = 0.25;

        // ---- front end (2026-10-01 bench: thesis_vio/scripts/frontend_bench.py) ----------
        // Every default reproduces the original behaviour, so an arm changes only what it sets.
        //
        // front_end:
        //   "gftt"       corners over the whole image, KLT, +-3 px depth lookup (original)
        //   "gftt_mask"  corners only where a depth lookup can succeed. ~300 corners per frame
        //                but a median of 18 got depth on t6 clear; masking: 74% -> 95% good.
        //   "sonar_klt"  track the projected sonar returns themselves: scene-anchored, exact
        //                depth, no lookup. ~100% good in clear water, 25 ms/pair.
        std::string front_end = "gftt";
        // Image used for tracking: "none" | "clahe" | "stretch" | "stretch_clahe". The quality
        // gate's sharpness/contrast are ALWAYS measured on the raw luminance, so enhancement
        // does not change what the visibility check sees. Semi-turbid: 8.5% -> 73.6% good.
        std::string enhance = "none";
        double clahe_clip = 2.0;
        // Which scan's sonar cloud supplies depth for the PREVIOUS frame's features:
        //   "current"  the scan being processed (original; the vehicle has moved ~1 scan since
        //              the previous image was taken)
        //   "previous" the scan that was current when the previous image was taken (correct)
        // sonar_klt always uses "previous": its seeds must come from the previous image.
        std::string depth_scan = "current";
        double sonar_seed_spacing_px = 3.0;   // one seed per grid cell, nearest return kept
        // Minimum Sobel magnitude at a seed (tracking image); 0 = keep every return. Keep 0:
        // in enhanced turbid frames the strongest gradients are 8-bit quantisation contours,
        // so filtering on gradient selects exactly the false edges (t6 semi, gyro-gated:
        // 74.5% good / 2.3% bad at 0 vs 61.3% / 3.2% at 8). The FB check removes flat seeds.
        double sonar_seed_min_gradient = 0.0;
        int sonar_seed_max = 1500;
        // Quality-gate check: reject when the PnP rotation disagrees with the integrated gyro
        // by more than this (degrees). 0 disables. Strongest bad-measurement predictor on the
        // bench (Spearman 0.73-0.97 vs ~0.5 for reprojection); at 0.3 it cost ~1% of good
        // clear-water measurements and cut semi bad measurements to ~3%.
        double gate_max_gyro_deg = 0.0;
    };

    struct Diagnostics
    {
        std::uint64_t frames_received = 0;
        std::uint64_t frames_processed = 0;
        std::uint64_t updates_applied = 0;
        std::uint64_t rejected_few_features = 0;
        std::uint64_t rejected_few_depth = 0;
        std::uint64_t rejected_pose_failed = 0;
        std::uint64_t rejected_implausible = 0;
        std::uint64_t rejected_innovation_gate = 0;
        // quality-gate reasons, counted separately so the dominant one is visible
        std::uint64_t rejected_q_inliers = 0;
        std::uint64_t rejected_q_ratio = 0;
        std::uint64_t rejected_q_reproj = 0;
        std::uint64_t rejected_q_image = 0;
        std::uint64_t rejected_q_speed = 0;
        std::uint64_t rejected_residual = 0;
        std::uint64_t rejected_q_gyro = 0;
        std::uint64_t rejected_stale = 0;     ///< image too old for the scan (stream out of sync)
        double last_image_age = 0.0;          ///< scan end - image stamp, seconds
        std::uint64_t gyro_unavailable = 0;   ///< gyro check skipped: buffer did not cover dt
        // Measurements that came out of PnP. Distinct from updates_applied, which now counts
        // only what actually reached the filter -- the two used to share one counter, so
        // every log line reported gate-rejected measurements as "applied".
        std::uint64_t measurements_produced = 0;

        int last_tracked_features = 0;
        int last_depth_features = 0;
        int last_inliers = 0;
        double last_inlier_ratio = 0.0;
        double last_reproj_px = 0.0;
        double last_translation_norm = 0.0;
        double last_residual_norm = 0.0;
        /// Speed the camera picks up purely from body rotation about its lever arm,
        /// |omega x p_cam|. Logged so the size of the correction can be read off real runs
        /// and correlated with the turn-time degradation seen in RViz.
        double last_lever_arm_speed = 0.0;
        /// Angle between the PnP rotation and the integrated-gyro rotation (deg); NaN when
        /// the gyro buffer did not cover the frame interval.
        double last_gyro_deg = std::numeric_limits<double>::quiet_NaN();

        // Recorded even though nothing acts on it yet: this is the signal the visibility
        // gate will eventually key on, and logging it now means the threshold can be
        // calibrated from real runs instead of guessed.
        double last_sharpness = 0.0;
        double last_contrast = 0.0;
    };

    VisualFusion() = default;

    void load_parameters(rclcpp::Node &node);
    bool enabled() const { return params_.enabled; }
    const Params &params() const { return params_; }
    const Diagnostics &diagnostics() const { return diagnostics_; }

    /// Buffer an incoming image. Cheap: decoding and tracking happen in process().
    void push_image(const sensor_msgs::msg::Image &msg);

    /// Buffer this scan's gyro samples (body frame), for the PnP-vs-gyro gate check. Call
    /// every scan BEFORE take_measurement(); duplicate stamps are ignored.
    void push_imu(const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> &imu)
    {
        if (!params_.enabled) return;
        for (const auto &m : imu)
        {
            if (!m) continue;
            gyro_.push(rclcpp::Time(m->header.stamp).seconds(),
                       V3D(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z));
        }
    }

    /// Attach the sonar points of the current scan, in sonar frame, for depth lookup.
    void set_depth_cloud(const PointCloudXYZI::Ptr &cloud, double stamp);

    /// Compute this scan's visual measurement (tracking + PnP). Call ONCE per scan,
    /// before the iterated update. Returns true when a usable measurement was produced.
    ///
    /// Split from the row assembly below because tracking and PnP are expensive and must
    /// run once, whereas the residual has to be recomputed against the state at every
    /// iEKF iteration -- the same division the DVL/pressure/mag path uses.
    bool take_measurement(double scan_end_time, const Ekf &kf, rclcpp::Node &node);

    /// Rows this measurement will contribute to the stacked Jacobian (3 or 0).

    /// Append the visual rows to the joint measurement Jacobian, exactly as
    /// AuxiliarySensorFusion::append_joint_measurement_rows does for the other sensors.
    /// Rows are whitened by 1/sigma so the solver treats every sensor consistently.
    /// Build the visual velocity measurement as (residual, H, R) for a standard linear
    /// update. Upstream applies each auxiliary sensor sequentially through
    /// AuxiliarySensorFusion::apply_external_update rather than stacking rows into the
    /// sonar solve, so this returns an unwhitened measurement and lets that path own the
    /// filter mathematics.
    ///
    /// \param raw_gyro Un-debiased body-frame angular rate, needed for the camera lever arm.
    /// \return false if there is no measurement, or the innovation gate rejected it.
    bool build_update(const state_ikfom &state,
                      const V3D &raw_gyro,
                      Eigen::VectorXd &residual_out,
                      Eigen::MatrixXd &H_out,
                      Eigen::MatrixXd &R_out);

    /// Discard the current measurement once the update has consumed it.
    void clear_measurement() { has_measurement_ = false; }

private:
    struct Frame
    {
        double stamp = 0.0;
        cv::Mat gray;                           ///< tracking image (after enhancement)
        visual_frontend::Projection projection; ///< sonar scan current when it was taken
    };

    /// Tracked-feature correspondences between the previous and current frame.
    struct TrackResult
    {
        std::vector<cv::Point2f> previous;
        std::vector<cv::Point2f> current;
        /// Camera-frame 3D point of each `previous` feature when the front end knows it
        /// exactly (sonar_klt); empty means "look the depth up".
        std::vector<cv::Point3f> previous_xyz;
    };

    /// raw_gray: luminance exactly as before (for the image-quality gate).
    /// track_gray: what the front end tracks on (equal to raw_gray when enhance == none).
    bool decode(const sensor_msgs::msg::Image &msg, cv::Mat &raw_gray, cv::Mat &track_gray) const;
    TrackResult track(const cv::Mat &previous_gray, const cv::Mat &current_gray,
                      const visual_frontend::Projection &depth_projection) const;

    /// Project the current sonar cloud into the (downscaled) image.
    visual_frontend::Projection project_current_cloud();
    visual_frontend::Intrinsics scaled_intrinsics() const;

    /// Recover camera motion from correspondences with known depth (PnP on 3D-2D pairs).
    bool estimate_motion(const TrackResult &tracks,
                         const visual_frontend::Projection &depth_projection,
                         Eigen::Isometry3d &T_prev_curr,
                         int &inliers);

    void compute_image_quality(const cv::Mat &gray);

    Params params_;
    Diagnostics diagnostics_;

    std::mutex image_mutex_;
    std::deque<sensor_msgs::msg::Image> pending_images_;

    std::mutex cloud_mutex_;
    PointCloudXYZI::Ptr depth_cloud_;
    double depth_cloud_stamp_ = -1.0;

    Frame previous_frame_;
    bool has_previous_frame_ = false;

    visual_frontend::Enhance enhance_mode_ = visual_frontend::Enhance::None;
    visual_frontend::GyroBuffer gyro_;

    // Measurement carried from take_measurement() to append_joint_measurement_rows().
    bool has_measurement_ = false;
    V3D measured_cam_velocity_ = V3D::Zero();  ///< camera-frame velocity, metres/second
    double measurement_dt_ = 0.0;
    double measurement_t_prev_ = 0.0;   ///< image stamps the measurement spans, for the CSV
    double measurement_t_curr_ = 0.0;
    std::ofstream debug_csv_;

    /// One row per measurement that came out of PnP, accepted or not. `pred` is null for
    /// rows rejected before the state was consulted (the quality gate needs no state).
    void write_csv(bool accepted, const char *reason, const V3D *pred, double residual)
    {
        if (!debug_csv_.is_open()) return;
        const V3D &v = measured_cam_velocity_;
        debug_csv_ << std::fixed << std::setprecision(6)
                   << measurement_t_prev_ << ',' << measurement_t_curr_ << ','
                   << measurement_dt_ << ',' << diagnostics_.last_tracked_features << ','
                   << diagnostics_.last_depth_features << ',' << diagnostics_.last_inliers << ','
                   << diagnostics_.last_inlier_ratio << ',' << diagnostics_.last_reproj_px << ','
                   << diagnostics_.last_sharpness << ',' << diagnostics_.last_contrast << ','
                   << v.x() << ',' << v.y() << ',' << v.z() << ',' << v.norm() << ',';
        if (pred) debug_csv_ << pred->x() << ',' << pred->y() << ',' << pred->z() << ',' << residual;
        else      debug_csv_ << ",,,";
        debug_csv_ << ',' << (accepted ? 1 : 0) << ',' << reason << ','
                   << diagnostics_.last_gyro_deg << '\n';
        // The node is stopped with SIGKILL, which never flushes the stream buffer.
        debug_csv_.flush();
    }
};


#include <algorithm>
#include <cmath>
#include <limits>

// cv_bridge dropped the .h header after Humble; Iron onwards ship only .hpp.
// __has_include keeps this compiling on both without a distro check.
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace {

/// Read a 9-element row-major matrix parameter, falling back to identity.
inline M3D matrix_parameter(rclcpp::Node &node, const std::string &name, const M3D &fallback)
{
    std::vector<double> values;
    node.get_parameter_or(name, values, std::vector<double>());
    if (values.size() != 9)
    {
        return fallback;
    }

    M3D matrix;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            matrix(row, col) = values[static_cast<std::size_t>(row * 3 + col)];
        }
    }
    return matrix;
}

inline V3D vector_parameter(rclcpp::Node &node, const std::string &name, const V3D &fallback)
{
    std::vector<double> values;
    node.get_parameter_or(name, values, std::vector<double>());
    if (values.size() != 3)
    {
        return fallback;
    }
    return V3D(values[0], values[1], values[2]);
}

}  // namespace


inline void VisualFusion::load_parameters(rclcpp::Node &node)
{
    node.get_parameter_or("visual.enabled", params_.enabled, false);

    node.get_parameter_or("visual.fx", params_.fx, params_.fx);
    node.get_parameter_or("visual.fy", params_.fy, params_.fy);
    node.get_parameter_or("visual.cx", params_.cx, params_.cx);
    node.get_parameter_or("visual.cy", params_.cy, params_.cy);
    node.get_parameter_or("visual.image_width", params_.image_width, params_.image_width);
    node.get_parameter_or("visual.image_height", params_.image_height, params_.image_height);

    params_.R_cam_sonar = matrix_parameter(node, "visual.extrinsic_R_cam_sonar",
                                           params_.R_cam_sonar);
    params_.t_cam_sonar = vector_parameter(node, "visual.extrinsic_T_cam_sonar",
                                           params_.t_cam_sonar);

    node.get_parameter_or("visual.max_features", params_.max_features, params_.max_features);
    node.get_parameter_or("visual.feature_quality", params_.feature_quality,
                          params_.feature_quality);
    node.get_parameter_or("visual.min_feature_distance", params_.min_feature_distance,
                          params_.min_feature_distance);
    node.get_parameter_or("visual.max_fb_error_px", params_.max_fb_error_px,
                          params_.max_fb_error_px);
    node.get_parameter_or("visual.min_tracked_features", params_.min_tracked_features,
                          params_.min_tracked_features);
    node.get_parameter_or("visual.min_depth_features", params_.min_depth_features,
                          params_.min_depth_features);
    node.get_parameter_or("visual.max_reprojection_error_px",
                          params_.max_reprojection_error_px,
                          params_.max_reprojection_error_px);
    node.get_parameter_or("visual.max_translation_per_frame_m",
                          params_.max_translation_per_frame_m,
                          params_.max_translation_per_frame_m);
    node.get_parameter_or("visual.position_cov", params_.position_cov, params_.position_cov);
    node.get_parameter_or("visual.innovation_gate_sigma", params_.innovation_gate_sigma,
                          params_.innovation_gate_sigma);
    node.get_parameter_or("visual.image_scale", params_.image_scale, params_.image_scale);

    // The gate parameters are declared here rather than in laserMapping.cpp so the module
    // stays self-contained. They MUST be declared: an undeclared parameter silently ignores
    // a -p override and returns the default, which would make an A/B run quietly test the
    // same thing twice.
    auto declared = [&node](const std::string &name, auto &value) {
        using T = std::decay_t<decltype(value)>;
        if (!node.has_parameter(name))
        {
            node.declare_parameter<T>(name, value);
        }
        node.get_parameter(name, value);
    };
    declared("visual.gate_mode", params_.gate_mode);
    declared("visual.gate_min_inliers", params_.gate_min_inliers);
    declared("visual.gate_min_inlier_ratio", params_.gate_min_inlier_ratio);
    declared("visual.gate_max_reproj_px", params_.gate_max_reproj_px);
    declared("visual.gate_min_sharpness", params_.gate_min_sharpness);
    declared("visual.gate_min_contrast", params_.gate_min_contrast);
    declared("visual.gate_max_speed", params_.gate_max_speed);
    declared("visual.gate_max_residual", params_.gate_max_residual);
    declared("visual.debug_csv", params_.debug_csv);
    // Previously hard-coded (never read from parameters); declared so they can be swept.
    declared("visual.klt_window", params_.klt_window);
    declared("visual.klt_pyramid_levels", params_.klt_pyramid_levels);
    declared("visual.front_end", params_.front_end);
    declared("visual.enhance", params_.enhance);
    declared("visual.clahe_clip", params_.clahe_clip);
    declared("visual.depth_scan", params_.depth_scan);
    declared("visual.sonar_seed_spacing_px", params_.sonar_seed_spacing_px);
    declared("visual.sonar_seed_min_gradient", params_.sonar_seed_min_gradient);
    declared("visual.sonar_seed_max", params_.sonar_seed_max);
    declared("visual.gate_max_gyro_deg", params_.gate_max_gyro_deg);
    declared("visual.max_image_age", params_.max_image_age);
    if (params_.front_end != "gftt" && params_.front_end != "gftt_mask" &&
        params_.front_end != "sonar_klt")
    {
        RCLCPP_WARN(node.get_logger(), "visual.front_end '%s' unknown, using 'gftt'",
                    params_.front_end.c_str());
        params_.front_end = "gftt";
    }
    if (const auto e = visual_frontend::parse_enhance(params_.enhance))
    {
        enhance_mode_ = *e;
    }
    else
    {
        RCLCPP_WARN(node.get_logger(), "visual.enhance '%s' unknown, using 'none'",
                    params_.enhance.c_str());
        params_.enhance = "none";
        enhance_mode_ = visual_frontend::Enhance::None;
    }
    if (params_.depth_scan != "current" && params_.depth_scan != "previous")
    {
        RCLCPP_WARN(node.get_logger(), "visual.depth_scan '%s' unknown, using 'current'",
                    params_.depth_scan.c_str());
        params_.depth_scan = "current";
    }
    if (params_.gate_mode != "quality" && params_.gate_mode != "innovation" &&
        params_.gate_mode != "none")
    {
        RCLCPP_WARN(node.get_logger(), "visual.gate_mode '%s' unknown, using 'quality'",
                    params_.gate_mode.c_str());
        params_.gate_mode = "quality";
    }
    if (params_.enabled && !params_.debug_csv.empty())
    {
        debug_csv_.open(params_.debug_csv);
        debug_csv_ << "t_prev,t_curr,dt,tracked,depth,inliers,inlier_ratio,reproj_px,"
                      "sharpness,contrast,vx,vy,vz,speed,px,py,pz,residual,accepted,reason,gyro_deg\n";
    }
    RCLCPP_INFO(node.get_logger(),
                "VisualFusion gate=%s  min_inliers=%d  min_ratio=%.2f  max_reproj=%.2f px  "
                "min_sharp=%.1f  min_contrast=%.1f  max_speed=%.2f  max_residual=%.2f  "
                "max_gyro=%.2f deg",
                params_.gate_mode.c_str(), params_.gate_min_inliers,
                params_.gate_min_inlier_ratio, params_.gate_max_reproj_px,
                params_.gate_min_sharpness, params_.gate_min_contrast,
                params_.gate_max_speed, params_.gate_max_residual, params_.gate_max_gyro_deg);
    RCLCPP_INFO(node.get_logger(),
                "VisualFusion front_end=%s  enhance=%s (clahe_clip %.1f)  depth_scan=%s  "
                "seeds: spacing %.1f px, min_grad %.1f, max %d  max_image_age=%.3f s",
                params_.front_end.c_str(), params_.enhance.c_str(), params_.clahe_clip,
                params_.depth_scan.c_str(), params_.sonar_seed_spacing_px,
                params_.sonar_seed_min_gradient, params_.sonar_seed_max, params_.max_image_age);
    node.get_parameter_or("visual.max_image_dt", params_.max_image_dt, params_.max_image_dt);

    params_.image_scale = std::clamp(params_.image_scale, 0.1, 1.0);

    RCLCPP_INFO(node.get_logger(),
                "VisualFusion %s (fx=%.1f fy=%.1f, %d features, scale %.2f)",
                params_.enabled ? "ENABLED" : "disabled",
                params_.fx, params_.fy, params_.max_features, params_.image_scale);
}

inline void VisualFusion::push_image(const sensor_msgs::msg::Image &msg)
{
    if (!params_.enabled)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(image_mutex_);
    diagnostics_.frames_received++;
    pending_images_.push_back(msg);

    // The camera can outrun the sonar scan rate; keeping a short queue bounds memory and
    // guarantees the frame used is close in time to the scan being processed.
    while (pending_images_.size() > 10)
    {
        pending_images_.pop_front();
    }
}

inline void VisualFusion::set_depth_cloud(const PointCloudXYZI::Ptr &cloud, double stamp)
{
    if (!params_.enabled)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(cloud_mutex_);
    depth_cloud_ = cloud;
    depth_cloud_stamp_ = stamp;
}

inline bool VisualFusion::decode(const sensor_msgs::msg::Image &msg, cv::Mat &raw_gray,
                                 cv::Mat &track_gray) const
{
    try
    {
        const cv_bridge::CvImageConstPtr bridge = cv_bridge::toCvShare(
            std::make_shared<sensor_msgs::msg::Image>(msg), msg.encoding);
        if (bridge->image.empty())
        {
            return false;
        }

        // Raw luminance, exactly as before: the image-quality gate keeps measuring this.
        cv::Mat mono;
        if (bridge->image.channels() == 3)
        {
            cv::cvtColor(bridge->image, mono, cv::COLOR_BGR2GRAY);
        }
        else if (bridge->image.channels() == 4)
        {
            cv::cvtColor(bridge->image, mono, cv::COLOR_BGRA2GRAY);
        }
        else
        {
            mono = bridge->image;
        }

        if (params_.image_scale < 0.999)
        {
            cv::resize(mono, raw_gray, cv::Size(), params_.image_scale, params_.image_scale,
                       cv::INTER_AREA);
        }
        else
        {
            raw_gray = mono.clone();
        }
        if (raw_gray.empty())
        {
            return false;
        }

        if (enhance_mode_ == visual_frontend::Enhance::None)
        {
            track_gray = raw_gray;
            return true;
        }
        // Enhancement needs the colour channels (the best one is picked per frame), so it
        // works from the colour image, downscaled the same way.
        cv::Mat colour;
        if (params_.image_scale < 0.999)
        {
            cv::resize(bridge->image, colour, cv::Size(), params_.image_scale,
                       params_.image_scale, cv::INTER_AREA);
        }
        else
        {
            colour = bridge->image;
        }
        track_gray = visual_frontend::enhance(colour, enhance_mode_, params_.clahe_clip);
        return !track_gray.empty();
    }
    catch (const std::exception &)
    {
        return false;
    }
}

inline void VisualFusion::compute_image_quality(const cv::Mat &gray)
{
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    diagnostics_.last_sharpness = stddev[0] * stddev[0];

    cv::meanStdDev(gray, mean, stddev);
    diagnostics_.last_contrast = stddev[0];
}

inline VisualFusion::TrackResult VisualFusion::track(
    const cv::Mat &previous_gray, const cv::Mat &current_gray,
    const visual_frontend::Projection &depth_projection) const
{
    TrackResult result;

    // Forward-backward KLT in both branches: track back to the original frame and keep only
    // features that return close to where they started. Turbidity produces plenty of
    // confident but wrong matches in the backscatter, and this rejects them cheaply.
    std::vector<cv::Point2f> previous_points;
    std::vector<cv::Point3f> previous_xyz;
    if (params_.front_end == "sonar_klt")
    {
        // Seeds are the previous scan's sonar returns as they project into the previous
        // image: every correspondence is on real structure and carries an exact depth.
        visual_frontend::select_sonar_seeds(depth_projection, previous_gray,
                                            params_.sonar_seed_spacing_px,
                                            params_.sonar_seed_min_gradient,
                                            params_.sonar_seed_max, previous_points, previous_xyz);
    }
    else
    {
        // gftt_mask: detect only where a depth lookup can succeed -- elsewhere a corner is
        // tracked and then thrown away for want of depth.
        const bool masked = params_.front_end == "gftt_mask" && !depth_projection.lookup_mask.empty();
        cv::goodFeaturesToTrack(previous_gray, previous_points, params_.max_features,
                                params_.feature_quality, params_.min_feature_distance,
                                masked ? depth_projection.lookup_mask : cv::noArray());
    }
    if (previous_points.empty())
    {
        return result;
    }

    std::vector<cv::Point2f> current_points;
    const std::vector<uchar> ok = visual_frontend::track_klt(
        previous_gray, current_gray, previous_points, current_points, params_.klt_window,
        params_.klt_pyramid_levels, params_.max_fb_error_px);

    for (std::size_t i = 0; i < previous_points.size(); ++i)
    {
        if (!ok[i])
        {
            continue;
        }
        result.previous.push_back(previous_points[i]);
        result.current.push_back(current_points[i]);
        if (!previous_xyz.empty())
        {
            result.previous_xyz.push_back(previous_xyz[i]);
        }
    }

    return result;
}

inline visual_frontend::Intrinsics VisualFusion::scaled_intrinsics() const
{
    const double s = params_.image_scale;
    return {params_.fx * s, params_.fy * s, params_.cx * s, params_.cy * s,
            static_cast<int>(params_.image_width * s), static_cast<int>(params_.image_height * s)};
}

inline visual_frontend::Projection VisualFusion::project_current_cloud()
{
    PointCloudXYZI::Ptr cloud;
    {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        cloud = depth_cloud_;
    }
    const visual_frontend::Intrinsics k = scaled_intrinsics();
    if (!cloud)
    {
        return visual_frontend::build_projection(std::vector<PointType>(),
                                                 params_.R_cam_sonar, params_.t_cam_sonar, k);
    }
    // Same rules as the original build_depth_lookup (nearest return per pixel), so the
    // legacy front end sees an identical depth map.
    return visual_frontend::build_projection(cloud->points, params_.R_cam_sonar,
                                             params_.t_cam_sonar, k);
}

inline bool VisualFusion::estimate_motion(const TrackResult &tracks,
                                   const visual_frontend::Projection &depth_projection,
                                   Eigen::Isometry3d &T_prev_curr,
                                   int &inliers)
{
    // Back-project the previous frame's features to 3D using sonar depth, then solve for
    // the camera pose that reprojects them onto their tracked positions in the current
    // frame. Depth from sonar is what makes the translation metric.
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;

    const visual_frontend::Intrinsics k = scaled_intrinsics();
    const double fx = k.fx;
    const double fy = k.fy;
    const double cx = k.cx;
    const double cy = k.cy;

    for (std::size_t i = 0; i < tracks.previous.size(); ++i)
    {
        if (!tracks.previous_xyz.empty())
        {
            // Exact: the feature IS a sonar return.
            object_points.push_back(tracks.previous_xyz[i]);
            image_points.push_back(tracks.current[i]);
            continue;
        }
        const std::optional<double> depth =
            visual_frontend::depth_at(depth_projection.depth, tracks.previous[i]);
        if (!depth.has_value())
        {
            continue;
        }

        const double z = depth.value();
        const double x = (tracks.previous[i].x - cx) * z / fx;
        const double y = (tracks.previous[i].y - cy) * z / fy;

        object_points.emplace_back(static_cast<float>(x), static_cast<float>(y),
                                   static_cast<float>(z));
        image_points.push_back(tracks.current[i]);
    }

    diagnostics_.last_depth_features = static_cast<int>(object_points.size());
    if (static_cast<int>(object_points.size()) < params_.min_depth_features)
    {
        diagnostics_.rejected_few_depth++;
        return false;
    }

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                                       0.0, fy, cy,
                                                       0.0, 0.0, 1.0);
    cv::Mat distortion = cv::Mat::zeros(4, 1, CV_64F);
    cv::Mat rvec;
    cv::Mat tvec;
    std::vector<int> inlier_indices;

    const bool solved = cv::solvePnPRansac(
        object_points, image_points, camera_matrix, distortion, rvec, tvec,
        false, 100, static_cast<float>(params_.max_reprojection_error_px), 0.99,
        inlier_indices, cv::SOLVEPNP_ITERATIVE);

    inliers = static_cast<int>(inlier_indices.size());
    if (!solved || inliers < params_.min_depth_features)
    {
        diagnostics_.rejected_pose_failed++;
        return false;
    }

    // Fit quality of the accepted solution: how well its own inliers reproject, and what share
    // of the depth-carrying features agreed with it. Both describe THIS measurement and are
    // independent of the filter state. Reprojection error is in the downscaled image.
    {
        std::vector<cv::Point3f> in_obj;
        std::vector<cv::Point2f> in_img;
        in_obj.reserve(inlier_indices.size());
        in_img.reserve(inlier_indices.size());
        for (int idx : inlier_indices)
        {
            in_obj.push_back(object_points[static_cast<std::size_t>(idx)]);
            in_img.push_back(image_points[static_cast<std::size_t>(idx)]);
        }
        std::vector<cv::Point2f> projected;
        cv::projectPoints(in_obj, rvec, tvec, camera_matrix, distortion, projected);
        double sum = 0.0;
        for (std::size_t k = 0; k < projected.size(); ++k)
        {
            sum += cv::norm(projected[k] - in_img[k]);
        }
        diagnostics_.last_reproj_px = projected.empty() ? 0.0 : sum / projected.size();
        diagnostics_.last_inlier_ratio =
            object_points.empty() ? 0.0 : static_cast<double>(inliers) / object_points.size();
    }

    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);

    M3D R;
    V3D t;
    for (int row = 0; row < 3; ++row)
    {
        t(row) = tvec.at<double>(row);
        for (int col = 0; col < 3; ++col)
        {
            R(row, col) = rotation.at<double>(row, col);
        }
    }

    // solvePnP returns the POINT transform:  X_curr = R * X_prev + t.
    //
    // That is NOT the camera's motion. If the camera displaces by d (expressed in the
    // previous camera frame), a static point satisfies X_curr = R (X_prev - d), so
    // t = -R d, i.e.  d = -R^T t.
    //
    // Using t directly yields a velocity of the correct MAGNITUDE pointing roughly 180 deg
    // the wrong way, which is why every magnitude-based check (max_translation_per_frame_m,
    // the norm diagnostics) passed it. Measured on the pipe bag with thesis_vio's bench:
    // direction error 176.45 deg before this fix, 3.66 deg after, with scale 0.997 either
    // way. It is the reason the visual arm never helped, degraded further as position_cov
    // was tightened, and tripped the innovation gate 102 times once that gate could fire.
    T_prev_curr = Eigen::Isometry3d::Identity();
    T_prev_curr.linear() = R;
    T_prev_curr.translation() = -R.transpose() * t;
    return true;
}

inline bool VisualFusion::take_measurement(double scan_end_time, const Ekf &kf, rclcpp::Node &node)
{
    if (!params_.enabled)
    {
        return false;
    }

    // Periodic status at INFO. The per-update line below is DEBUG, which is off by
    // default, so without this the module is completely silent whether it is working or
    // rejecting every frame -- and those two cases look identical from outside.
    RCLCPP_INFO_THROTTLE(
        node.get_logger(), *node.get_clock(), 5000,
        "visual[%s]: rx=%lu proc=%lu produced=%lu applied=%lu | rejects: feat=%lu depth=%lu "
        "pnp=%lu jump=%lu gate=%lu q_inl=%lu q_ratio=%lu q_reproj=%lu q_img=%lu q_speed=%lu "
        "resid=%lu gyro=%lu (n/a %lu) stale=%lu | last: track=%d depth=%d inl=%d reproj=%.2f "
        "sharp=%.1f contrast=%.1f gyro=%.2fdeg age=%.3fs",
        params_.gate_mode.c_str(),
        static_cast<unsigned long>(diagnostics_.frames_received),
        static_cast<unsigned long>(diagnostics_.frames_processed),
        static_cast<unsigned long>(diagnostics_.measurements_produced),
        static_cast<unsigned long>(diagnostics_.updates_applied),
        static_cast<unsigned long>(diagnostics_.rejected_few_features),
        static_cast<unsigned long>(diagnostics_.rejected_few_depth),
        static_cast<unsigned long>(diagnostics_.rejected_pose_failed),
        static_cast<unsigned long>(diagnostics_.rejected_implausible),
        static_cast<unsigned long>(diagnostics_.rejected_innovation_gate),
        static_cast<unsigned long>(diagnostics_.rejected_q_inliers),
        static_cast<unsigned long>(diagnostics_.rejected_q_ratio),
        static_cast<unsigned long>(diagnostics_.rejected_q_reproj),
        static_cast<unsigned long>(diagnostics_.rejected_q_image),
        static_cast<unsigned long>(diagnostics_.rejected_q_speed),
        static_cast<unsigned long>(diagnostics_.rejected_residual),
        static_cast<unsigned long>(diagnostics_.rejected_q_gyro),
        static_cast<unsigned long>(diagnostics_.gyro_unavailable),
        static_cast<unsigned long>(diagnostics_.rejected_stale),
        diagnostics_.last_tracked_features, diagnostics_.last_depth_features,
        diagnostics_.last_inliers, diagnostics_.last_reproj_px, diagnostics_.last_sharpness,
        diagnostics_.last_contrast, diagnostics_.last_gyro_deg, diagnostics_.last_image_age);

    // Take the newest frame at or before the end of this scan, so the visual measurement
    // refers to the same instant the filter state has just been propagated to.
    sensor_msgs::msg::Image selected;
    bool have_image = false;
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        while (!pending_images_.empty())
        {
            const double stamp = rclcpp::Time(pending_images_.front().header.stamp).seconds();
            if (stamp > scan_end_time)
            {
                break;
            }
            selected = pending_images_.front();
            have_image = true;
            pending_images_.pop_front();
        }
    }

    if (!have_image)
    {
        return false;
    }

    // A camera stream that lags the sonar still delivers images "at or before" every scan, so
    // the selection above happily pairs a scan with a seconds-old image. Refuse it loudly.
    {
        const double image_stamp = rclcpp::Time(selected.header.stamp).seconds();
        diagnostics_.last_image_age = scan_end_time - image_stamp;
        if (params_.max_image_age > 0.0 && diagnostics_.last_image_age > params_.max_image_age)
        {
            diagnostics_.rejected_stale++;
            RCLCPP_WARN_THROTTLE(node.get_logger(), *node.get_clock(), 5000,
                                 "visual: newest image is %.3f s older than the scan (limit %.3f s) "
                                 "-- camera stream out of sync with sonar; %lu images rejected",
                                 diagnostics_.last_image_age, params_.max_image_age,
                                 static_cast<unsigned long>(diagnostics_.rejected_stale));
            return false;
        }
    }

    const double stamp = rclcpp::Time(selected.header.stamp).seconds();
    cv::Mat raw_gray;
    cv::Mat gray;  // tracking image
    if (!decode(selected, raw_gray, gray))
    {
        return false;
    }

    diagnostics_.frames_processed++;
    compute_image_quality(raw_gray);

    // This scan's cloud, projected into this image. Kept with the frame so that, next scan,
    // the previous image's features can be given the depth of the scan they were seen with.
    visual_frontend::Projection current_projection = project_current_cloud();

    if (!has_previous_frame_)
    {
        previous_frame_.stamp = stamp;
        previous_frame_.gray = gray;
        previous_frame_.projection = std::move(current_projection);
        has_previous_frame_ = true;
        return false;
    }

    const double dt = stamp - previous_frame_.stamp;
    if (dt <= 0.0 || dt > params_.max_image_dt)
    {
        // Too stale to relate: restart tracking from this frame rather than fabricating a
        // motion estimate across the gap.
        previous_frame_.stamp = stamp;
        previous_frame_.gray = gray;
        previous_frame_.projection = std::move(current_projection);
        return false;
    }

    const bool use_previous_depth =
        params_.front_end == "sonar_klt" || params_.depth_scan == "previous";
    const Frame previous = previous_frame_;  // cheap: cv::Mat headers are shared
    const visual_frontend::Projection &depth_projection =
        use_previous_depth ? previous.projection : current_projection;

    const TrackResult tracks = track(previous.gray, gray, depth_projection);
    diagnostics_.last_tracked_features = static_cast<int>(tracks.previous.size());

    previous_frame_.stamp = stamp;
    previous_frame_.gray = gray;
    previous_frame_.projection = current_projection;

    if (static_cast<int>(tracks.previous.size()) < params_.min_tracked_features)
    {
        diagnostics_.rejected_few_features++;
        return false;
    }

    Eigen::Isometry3d T_prev_curr = Eigen::Isometry3d::Identity();
    int inliers = 0;
    if (!estimate_motion(tracks, depth_projection, T_prev_curr, inliers))
    {
        return false;
    }
    diagnostics_.last_inliers = inliers;

    // PnP rotation vs the gyro integrated over the same interval, both as camera point
    // transforms. Needs only the gyro bias and the camera<-body extrinsic from the state, not
    // its attitude or velocity, so it says nothing about whether the FILTER is lost.
    diagnostics_.last_gyro_deg = std::numeric_limits<double>::quiet_NaN();
    {
        const state_ikfom x = kf.get_x();
        const V3D bg(x.bg[0], x.bg[1], x.bg[2]);
        if (const auto R_b0_b1 = gyro_.integrate(previous.stamp, stamp, bg))
        {
            const M3D R_cam_body = params_.R_cam_sonar * x.offset_R_L_I.toRotationMatrix().transpose();
            const M3D R_gyro = visual_frontend::camera_rotation_from_body(*R_b0_b1, R_cam_body);
            diagnostics_.last_gyro_deg =
                visual_frontend::rotation_angle_deg(T_prev_curr.linear() * R_gyro.transpose());
        }
        else
        {
            diagnostics_.gyro_unavailable++;
        }
    }

    const V3D translation_cam = T_prev_curr.translation();
    diagnostics_.last_translation_norm = translation_cam.norm();

    if (translation_cam.norm() > params_.max_translation_per_frame_m)
    {
        // A gross visual failure shows up as an implausible jump well before it shows up
        // as a large innovation, so this catches what the gate below would miss.
        diagnostics_.rejected_implausible++;
        return false;
    }

    // Store the measurement as a CAMERA-FRAME VELOCITY.
    //
    // Dividing the inter-frame displacement by dt makes this dimensionally a velocity,
    // which is what the residual below actually compares against. The previous version
    // kept a displacement but wrote the Jacobian into the POSITION block, telling the
    // filter it had observed absolute position when it had observed a per-frame motion --
    // and the IMU propagation had already moved position using that same velocity, so the
    // correction partly double-counted. Treating it as a velocity, exactly as the DVL
    // does, removes that inconsistency.
    measured_cam_velocity_ = translation_cam / dt;
    measurement_dt_ = dt;
    measurement_t_prev_ = stamp - dt;
    measurement_t_curr_ = stamp;
    diagnostics_.measurements_produced++;

    // Quality gate. Everything here is a property of the measurement, none of it of the
    // filter state, so it gives the same verdict whether the filter is on track or lost.
    if (params_.gate_mode == "quality")
    {
        const Diagnostics &d = diagnostics_;
        const char *reason = nullptr;
        if (inliers < params_.gate_min_inliers)
        {
            diagnostics_.rejected_q_inliers++; reason = "inliers";
        }
        else if (params_.gate_min_inlier_ratio > 0.0 && d.last_inlier_ratio < params_.gate_min_inlier_ratio)
        {
            diagnostics_.rejected_q_ratio++; reason = "ratio";
        }
        else if (params_.gate_max_reproj_px > 0.0 && d.last_reproj_px > params_.gate_max_reproj_px)
        {
            diagnostics_.rejected_q_reproj++; reason = "reproj";
        }
        else if ((params_.gate_min_sharpness > 0.0 && d.last_sharpness < params_.gate_min_sharpness) ||
                 (params_.gate_min_contrast > 0.0 && d.last_contrast < params_.gate_min_contrast))
        {
            diagnostics_.rejected_q_image++; reason = "image";
        }
        else if (params_.gate_max_speed > 0.0 && measured_cam_velocity_.norm() > params_.gate_max_speed)
        {
            diagnostics_.rejected_q_speed++; reason = "speed";
        }
        else if (params_.gate_max_gyro_deg > 0.0 && std::isfinite(d.last_gyro_deg) &&
                 d.last_gyro_deg > params_.gate_max_gyro_deg)
        {
            diagnostics_.rejected_q_gyro++; reason = "gyro";
        }
        if (reason)
        {
            write_csv(false, reason, nullptr, 0.0);
            return false;
        }
    }
    has_measurement_ = true;

    RCLCPP_DEBUG(node.get_logger(),
                 "visual measurement: %d tracked, %d with depth, %d inliers, |v|=%.3f m/s",
                 diagnostics_.last_tracked_features, diagnostics_.last_depth_features,
                 inliers, measured_cam_velocity_.norm());

    return true;
}

inline bool VisualFusion::build_update(const state_ikfom &state,
                                       const V3D &raw_gyro,
                                       Eigen::VectorXd &residual_out,
                                       Eigen::MatrixXd &H_out,
                                       Eigen::MatrixXd &R_out)
{
    if (!has_measurement_)
    {
        return false;
    }

    // Predict the camera-frame velocity from the current state iterate.
    //
    //   v_cam = R_cam_body * ( R_world_body^T * v_world  +  omega_body x p_cam )
    //
    // built from the calibrated camera<-sonar rotation, the sonar<-IMU extrinsic that
    // lives in the filter state, and the current attitude. This mirrors the DVL, which
    // predicts a sensor-frame velocity from the same world velocity over its own lever arm.
    const M3D R_wb = state.rot.toRotationMatrix();
    const M3D R_body_sonar = state.offset_R_L_I.toRotationMatrix();
    // Maps a body/IMU-frame vector into the camera optical frame.
    const M3D R_cam_body = params_.R_cam_sonar * R_body_sonar.transpose();
    const M3D A = R_cam_body * R_wb.transpose();

    // Camera origin in the IMU/body frame. Derived from extrinsics already in use rather
    // than added as a new calibration: offset_T_L_I places the sonar origin in the body
    // frame, and -R_cam_sonar^T * t_cam_sonar is the camera origin expressed in the sonar
    // frame. This reproduces the simulator's own [0.2, 0, 0.2] to under a micrometre, and
    // tracks offset_*_L_I if the sonar extrinsic is ever estimated rather than fixed.
    const V3D p_cam_body =
        state.offset_T_L_I - R_body_sonar * params_.R_cam_sonar.transpose() * params_.t_cam_sonar;

    const V3D v_world(state.vel[0], state.vel[1], state.vel[2]);
    const V3D v_imu_body = R_wb.transpose() * v_world;
    const V3D omega_body = raw_gyro - V3D(state.bg[0], state.bg[1], state.bg[2]);

    // PnP measures the CAMERA's displacement, and the camera sits 0.283 m off the IMU
    // origin -- within 5 mm of the DVL's own 0.288 m lever arm. Body rotation therefore
    // translates the camera even when the IMU origin is stationary. Omitting this term left
    // a bias proportional to |omega|, i.e. one that appears precisely during turns: about
    // 0.099 m/s at 20 deg/s, 15% of the 0.679 m/s cruise speed. Down-weighting does not
    // protect against that, because it is a bias and not noise.
    const V3D v_cam_pred = R_cam_body * (v_imu_body + omega_body.cross(p_cam_body));

    const V3D residual = measured_cam_velocity_ - v_cam_pred;
    diagnostics_.last_residual_norm = residual.norm();
    diagnostics_.last_lever_arm_speed = (R_cam_body * omega_body.cross(p_cam_body)).norm();

    // The same three blocks the DVL fills. Previously only the velocity block was set,
    // which told the filter the residual carried no attitude or gyro-bias information and
    // forced the entire innovation onto velocity.
    //
    // IKFoM applies a right attitude perturbation, R_true = R Exp(dtheta^), so
    // d(R^T v)/d(dtheta) = skew(R^T v). The lever-arm term does not depend on attitude.
    // For the gyro bias, omega = raw_gyro - bg gives d(omega x p)/d(bg) = skew(p).
    // H is the derivative of h(x), paired with the innovation z - h(x).
    Eigen::MatrixXd H_vis = Eigen::MatrixXd::Zero(3, state_ikfom::DOF);
    H_vis.block<3, 3>(0, 3) = R_cam_body * skew_sym_mat<double>(v_imu_body);
    H_vis.block<3, 3>(0, 12) = A;
    H_vis.block<3, 3>(0, 15) = R_cam_body * skew_sym_mat<double>(p_cam_body);
    // Frame-to-frame displacement carries no information about absolute position, so the
    // position block stays zero. Changing that needs a map-anchored measurement, not a
    // different Jacobian.

    // position_cov is specified as a per-frame displacement variance, so convert it to a
    // velocity variance. Dividing by dt^2 means a shorter frame interval yields a noisier
    // velocity from the same pixel-level uncertainty, which is the correct behaviour.
    const double dt2 = std::max(1e-6, measurement_dt_ * measurement_dt_);
    const double cov = params_.position_cov / dt2;

    // Reject the WHOLE measurement rather than individual components. Dropping one axis and
    // keeping the other two applies a partial correction from an estimate already known to
    // disagree with the state -- and a systematically wrong measurement applied repeatedly
    // drags the filter even when it is heavily down-weighted.
    if (params_.gate_mode == "innovation" && params_.innovation_gate_sigma > 0.0)
    {
        // Legacy gate, kept for A/B only. Its threshold is derived from position_cov, so it
        // changes whenever the update weight does.
        const double limit = params_.innovation_gate_sigma * std::sqrt(cov);
        for (int i = 0; i < 3; ++i)
        {
            if (std::abs(residual[i]) > limit)
            {
                diagnostics_.rejected_innovation_gate++;
                write_csv(false, "innovation", &v_cam_pred, residual.norm());
                return false;
            }
        }
    }
    else if (params_.gate_mode == "quality" && params_.gate_max_residual > 0.0 &&
             residual.norm() > params_.gate_max_residual)
    {
        diagnostics_.rejected_residual++;
        write_csv(false, "residual", &v_cam_pred, residual.norm());
        return false;
    }

    // Counted HERE, where the measurement is actually handed to the filter. It used to be
    // counted when the measurement was produced, so gate rejections were reported as applied.
    diagnostics_.updates_applied++;
    write_csv(true, "", &v_cam_pred, residual.norm());

    residual_out = residual;
    H_out = H_vis;
    R_out = Eigen::MatrixXd::Identity(3, 3) * cov;
    return true;
}

#endif  // VISUAL_FUSION_HPP
