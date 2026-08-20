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
#include <optional>
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

#include "common_lib.h"
#include "use-ikfom.hpp"

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

        // Downscale before tracking. The 1920x1080 underwater camera is far higher
        // resolution than feature tracking needs, and the sonar cloud is sparse enough
        // that sub-pixel precision is not the limiting factor.
        double image_scale = 0.5;

        double max_image_dt = 0.25;      // seconds between usable frames
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

        int last_tracked_features = 0;
        int last_depth_features = 0;
        int last_inliers = 0;
        double last_translation_norm = 0.0;
        double last_residual_norm = 0.0;
        /// Speed the camera picks up purely from body rotation about its lever arm,
        /// |omega x p_cam|. Logged so the size of the correction can be read off real runs
        /// and correlated with the turn-time degradation seen in RViz.
        double last_lever_arm_speed = 0.0;

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
        cv::Mat gray;
    };

    /// Tracked-feature correspondences between the previous and current frame.
    struct TrackResult
    {
        std::vector<cv::Point2f> previous;
        std::vector<cv::Point2f> current;
    };

    bool decode(const sensor_msgs::msg::Image &msg, cv::Mat &gray) const;
    TrackResult track(const cv::Mat &previous_gray, const cv::Mat &current_gray) const;

    /// Build a sparse depth image by projecting the sonar cloud into the camera.
    void build_depth_lookup();
    std::optional<double> depth_at(const cv::Point2f &pixel) const;

    /// Recover camera motion from correspondences with known depth (PnP on 3D-2D pairs).
    bool estimate_motion(const TrackResult &tracks,
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

    // Sparse depth map in downscaled image coordinates; NaN where no sonar point projects.
    cv::Mat depth_lookup_;
    double depth_lookup_scale_ = 1.0;

    // Measurement carried from take_measurement() to append_joint_measurement_rows().
    bool has_measurement_ = false;
    V3D measured_cam_velocity_ = V3D::Zero();  ///< camera-frame velocity, metres/second
    double measurement_dt_ = 0.0;
};


#include <algorithm>
#include <cmath>
#include <limits>

#include <cv_bridge/cv_bridge.h>

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

inline bool VisualFusion::decode(const sensor_msgs::msg::Image &msg, cv::Mat &gray) const
{
    try
    {
        const cv_bridge::CvImageConstPtr bridge = cv_bridge::toCvShare(
            std::make_shared<sensor_msgs::msg::Image>(msg), msg.encoding);
        if (bridge->image.empty())
        {
            return false;
        }

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
            cv::resize(mono, gray, cv::Size(), params_.image_scale, params_.image_scale,
                       cv::INTER_AREA);
        }
        else
        {
            gray = mono.clone();
        }
        return !gray.empty();
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

inline VisualFusion::TrackResult VisualFusion::track(const cv::Mat &previous_gray,
                                              const cv::Mat &current_gray) const
{
    TrackResult result;

    std::vector<cv::Point2f> previous_points;
    cv::goodFeaturesToTrack(previous_gray, previous_points, params_.max_features,
                            params_.feature_quality, params_.min_feature_distance);
    if (previous_points.empty())
    {
        return result;
    }

    const cv::Size window(params_.klt_window, params_.klt_window);
    std::vector<cv::Point2f> current_points;
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, current_points,
                             status, error, window, params_.klt_pyramid_levels);

    // Forward-backward check: track back to the original frame and keep only features
    // that return close to where they started. Turbidity produces plenty of confident but
    // wrong matches in the backscatter, and this rejects them cheaply.
    std::vector<cv::Point2f> reverse_points;
    std::vector<uchar> reverse_status;
    std::vector<float> reverse_error;
    cv::calcOpticalFlowPyrLK(current_gray, previous_gray, current_points, reverse_points,
                             reverse_status, reverse_error, window,
                             params_.klt_pyramid_levels);

    for (std::size_t i = 0; i < previous_points.size(); ++i)
    {
        if (!status[i] || !reverse_status[i])
        {
            continue;
        }
        if (cv::norm(previous_points[i] - reverse_points[i]) > params_.max_fb_error_px)
        {
            continue;
        }
        result.previous.push_back(previous_points[i]);
        result.current.push_back(current_points[i]);
    }

    return result;
}

inline void VisualFusion::build_depth_lookup()
{
    PointCloudXYZI::Ptr cloud;
    {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        cloud = depth_cloud_;
    }

    depth_lookup_scale_ = params_.image_scale;
    const int width = static_cast<int>(params_.image_width * depth_lookup_scale_);
    const int height = static_cast<int>(params_.image_height * depth_lookup_scale_);
    if (width <= 0 || height <= 0)
    {
        depth_lookup_ = cv::Mat();
        return;
    }

    depth_lookup_ = cv::Mat(height, width, CV_32F,
                            cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    if (!cloud || cloud->empty())
    {
        return;
    }

    const double fx = params_.fx * depth_lookup_scale_;
    const double fy = params_.fy * depth_lookup_scale_;
    const double cx = params_.cx * depth_lookup_scale_;
    const double cy = params_.cy * depth_lookup_scale_;

    for (const auto &point : cloud->points)
    {
        const V3D sonar_point(point.x, point.y, point.z);
        const V3D camera_point = params_.R_cam_sonar * sonar_point + params_.t_cam_sonar;

        // Behind the image plane, or so close the projection is numerically unstable.
        if (camera_point.z() < 1e-3)
        {
            continue;
        }

        const int u = static_cast<int>(std::lround(fx * camera_point.x() / camera_point.z() + cx));
        const int v = static_cast<int>(std::lround(fy * camera_point.y() / camera_point.z() + cy));
        if (u < 0 || u >= width || v < 0 || v >= height)
        {
            continue;
        }

        // Keep the nearest point per pixel: a farther surface seen through the same ray is
        // occluded, and using it would place the feature at the wrong depth.
        float &stored = depth_lookup_.at<float>(v, u);
        const float candidate = static_cast<float>(camera_point.z());
        if (std::isnan(stored) || candidate < stored)
        {
            stored = candidate;
        }
    }
}

inline std::optional<double> VisualFusion::depth_at(const cv::Point2f &pixel) const
{
    if (depth_lookup_.empty())
    {
        return std::nullopt;
    }

    const int u = static_cast<int>(std::lround(pixel.x));
    const int v = static_cast<int>(std::lround(pixel.y));

    // Sonar returns are sparse in image space, so an exact pixel hit is unlikely. Search a
    // small neighbourhood and take the nearest valid depth.
    constexpr int kSearchRadius = 3;
    double best_depth = 0.0;
    double best_distance = std::numeric_limits<double>::max();

    for (int dv = -kSearchRadius; dv <= kSearchRadius; ++dv)
    {
        const int row = v + dv;
        if (row < 0 || row >= depth_lookup_.rows)
        {
            continue;
        }
        for (int du = -kSearchRadius; du <= kSearchRadius; ++du)
        {
            const int col = u + du;
            if (col < 0 || col >= depth_lookup_.cols)
            {
                continue;
            }
            const float value = depth_lookup_.at<float>(row, col);
            if (std::isnan(value))
            {
                continue;
            }
            const double distance = std::sqrt(static_cast<double>(du * du + dv * dv));
            if (distance < best_distance)
            {
                best_distance = distance;
                best_depth = value;
            }
        }
    }

    if (best_distance == std::numeric_limits<double>::max())
    {
        return std::nullopt;
    }
    return best_depth;
}

inline bool VisualFusion::estimate_motion(const TrackResult &tracks,
                                   Eigen::Isometry3d &T_prev_curr,
                                   int &inliers)
{
    // Back-project the previous frame's features to 3D using sonar depth, then solve for
    // the camera pose that reprojects them onto their tracked positions in the current
    // frame. Depth from sonar is what makes the translation metric.
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;

    const double fx = params_.fx * depth_lookup_scale_;
    const double fy = params_.fy * depth_lookup_scale_;
    const double cx = params_.cx * depth_lookup_scale_;
    const double cy = params_.cy * depth_lookup_scale_;

    for (std::size_t i = 0; i < tracks.previous.size(); ++i)
    {
        const std::optional<double> depth = depth_at(tracks.previous[i]);
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
        "visual: rx=%lu proc=%lu applied=%lu | rejects: feat=%lu depth=%lu pnp=%lu "
        "jump=%lu gate=%lu | last: track=%d depth=%d inl=%d sharp=%.1f contrast=%.1f",
        static_cast<unsigned long>(diagnostics_.frames_received),
        static_cast<unsigned long>(diagnostics_.frames_processed),
        static_cast<unsigned long>(diagnostics_.updates_applied),
        static_cast<unsigned long>(diagnostics_.rejected_few_features),
        static_cast<unsigned long>(diagnostics_.rejected_few_depth),
        static_cast<unsigned long>(diagnostics_.rejected_pose_failed),
        static_cast<unsigned long>(diagnostics_.rejected_implausible),
        static_cast<unsigned long>(diagnostics_.rejected_innovation_gate),
        diagnostics_.last_tracked_features, diagnostics_.last_depth_features,
        diagnostics_.last_inliers, diagnostics_.last_sharpness,
        diagnostics_.last_contrast);

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

    const double stamp = rclcpp::Time(selected.header.stamp).seconds();
    cv::Mat gray;
    if (!decode(selected, gray))
    {
        return false;
    }

    diagnostics_.frames_processed++;
    compute_image_quality(gray);

    if (!has_previous_frame_)
    {
        previous_frame_.stamp = stamp;
        previous_frame_.gray = gray;
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
        return false;
    }

    const TrackResult tracks = track(previous_frame_.gray, gray);
    diagnostics_.last_tracked_features = static_cast<int>(tracks.previous.size());

    previous_frame_.stamp = stamp;
    const cv::Mat previous_gray_for_depth = previous_frame_.gray;
    previous_frame_.gray = gray;

    if (static_cast<int>(tracks.previous.size()) < params_.min_tracked_features)
    {
        diagnostics_.rejected_few_features++;
        return false;
    }

    build_depth_lookup();

    Eigen::Isometry3d T_prev_curr = Eigen::Isometry3d::Identity();
    int inliers = 0;
    if (!estimate_motion(tracks, T_prev_curr, inliers))
    {
        return false;
    }
    diagnostics_.last_inliers = inliers;

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
    has_measurement_ = true;

    diagnostics_.updates_applied++;

    RCLCPP_DEBUG(node.get_logger(),
                 "visual measurement: %d tracked, %d with depth, %d inliers, |v|=%.3f m/s",
                 diagnostics_.last_tracked_features, diagnostics_.last_depth_features,
                 inliers, measured_cam_velocity_.norm());

    (void)previous_gray_for_depth;
    (void)kf;
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
    if (params_.innovation_gate_sigma > 0.0)
    {
        const double limit = params_.innovation_gate_sigma * std::sqrt(cov);
        for (int i = 0; i < 3; ++i)
        {
            if (std::abs(residual[i]) > limit)
            {
                diagnostics_.rejected_innovation_gate++;
                return false;
            }
        }
    }

    residual_out = residual;
    H_out = H_vis;
    R_out = Eigen::MatrixXd::Identity(3, 3) * cov;
    return true;
}

#endif  // VISUAL_FUSION_HPP
