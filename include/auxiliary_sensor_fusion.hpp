#ifndef AUXILIARY_SENSOR_FUSION_HPP
#define AUXILIARY_SENSOR_FUSION_HPP

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>

#include "common_lib.h"
#include "dvl_measurement_model.hpp"
#include "magnetometer_heading_model.hpp"
#include "use-ikfom.hpp"

class AuxiliarySensorFusion
{
public:
    using Ekf = esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom>;
    using DvlMsg = geometry_msgs::msg::TwistWithCovarianceStamped;
    using PressureMsg = sensor_msgs::msg::FluidPressure;
    using MagMsg = sensor_msgs::msg::MagneticField;

    struct UpdateSummary
    {
        int dvl_count = 0;
        int dvl_accepted = 0;
        int dvl_rejected = 0;
        int pressure_count = 0;
        int pressure_accepted = 0;
        int pressure_rejected = 0;
        int mag_count = 0;
        int mag_accepted = 0;
        int mag_rejected = 0;
        double dvl_res_norm_sum = 0.0;
        double dvl_res_norm_max = 0.0;
        V3D dvl_res_sum = V3D::Zero();
        V3D dvl_res_abs_max = V3D::Zero();
        V3D dvl_meas_sum = V3D::Zero();
        V3D dvl_pred_sum = V3D::Zero();
        V3D dvl_body_vel_sum = V3D::Zero();
        double pressure_res_depth_sum = 0.0;
        double pressure_res_depth_max = 0.0;
        double mag_res_norm_sum = 0.0;
        double mag_res_norm_max = 0.0;
        bool dvl_updated = false;
        bool pressure_updated = false;
        bool mag_updated = false;

        bool updated() const
        {
            return dvl_updated || pressure_updated || mag_updated;
        }

        double mean_dvl_residual() const
        {
            return dvl_count > 0 ? dvl_res_norm_sum / static_cast<double>(dvl_count) : 0.0;
        }

        V3D mean_dvl_residual_axis() const
        {
            if (dvl_count <= 0)
            {
                return V3D::Zero();
            }
            return dvl_res_sum / static_cast<double>(dvl_count);
        }

        V3D mean_dvl_measurement() const
        {
            if (dvl_count <= 0)
            {
                return V3D::Zero();
            }
            return dvl_meas_sum / static_cast<double>(dvl_count);
        }

        V3D mean_dvl_prediction() const
        {
            if (dvl_count <= 0)
            {
                return V3D::Zero();
            }
            return dvl_pred_sum / static_cast<double>(dvl_count);
        }

        V3D mean_dvl_body_velocity() const
        {
            if (dvl_count <= 0)
            {
                return V3D::Zero();
            }
            return dvl_body_vel_sum / static_cast<double>(dvl_count);
        }

        double mean_pressure_depth_residual() const
        {
            return pressure_count > 0 ? pressure_res_depth_sum / static_cast<double>(pressure_count) : 0.0;
        }

        double mean_mag_residual() const
        {
            return mag_count > 0 ? mag_res_norm_sum / static_cast<double>(mag_count) : 0.0;
        }

        void merge(const UpdateSummary &other)
        {
            dvl_count += other.dvl_count;
            dvl_accepted += other.dvl_accepted;
            dvl_rejected += other.dvl_rejected;
            pressure_count += other.pressure_count;
            pressure_accepted += other.pressure_accepted;
            pressure_rejected += other.pressure_rejected;
            mag_count += other.mag_count;
            mag_accepted += other.mag_accepted;
            mag_rejected += other.mag_rejected;
            dvl_res_norm_sum += other.dvl_res_norm_sum;
            dvl_res_norm_max = std::max(dvl_res_norm_max, other.dvl_res_norm_max);
            dvl_res_sum += other.dvl_res_sum;
            dvl_res_abs_max = dvl_res_abs_max.cwiseMax(other.dvl_res_abs_max);
            dvl_meas_sum += other.dvl_meas_sum;
            dvl_pred_sum += other.dvl_pred_sum;
            dvl_body_vel_sum += other.dvl_body_vel_sum;
            pressure_res_depth_sum += other.pressure_res_depth_sum;
            pressure_res_depth_max = std::max(pressure_res_depth_max, other.pressure_res_depth_max);
            mag_res_norm_sum += other.mag_res_norm_sum;
            mag_res_norm_max = std::max(mag_res_norm_max, other.mag_res_norm_max);
            dvl_updated = dvl_updated || other.dvl_updated;
            pressure_updated = pressure_updated || other.pressure_updated;
            mag_updated = mag_updated || other.mag_updated;
        }
    };

    enum class MeasurementKind
    {
        Dvl,
        Pressure,
        Magnetometer
    };

    struct TimedMeasurement
    {
        double timestamp = 0.0;
        MeasurementKind kind = MeasurementKind::Dvl;
        DvlMsg::ConstSharedPtr dvl;
        PressureMsg::ConstSharedPtr pressure;
        MagMsg::ConstSharedPtr magnetometer;
    };

    struct LateMeasurementCounts
    {
        std::size_t dvl = 0;
        std::size_t pressure = 0;
        std::size_t magnetometer = 0;

        std::size_t total() const
        {
            return dvl + pressure + magnetometer;
        }
    };

    void declare_parameters(rclcpp::Node &node)
    {
        node.declare_parameter<bool>("dvl.enable", false);
        node.declare_parameter<std::string>("dvl.topic", "/auv/dvl");
        node.declare_parameter<std::vector<double>>("dvl.extrinsic_T", {-0.079, -0.09691, -0.25938});
        node.declare_parameter<std::vector<double>>("dvl.extrinsic_R",
                                                    {1.0, 0.0, 0.0,
                                                     0.0, 1.0, 0.0,
                                                     0.0, 0.0, 1.0});
        node.declare_parameter<double>("dvl.dvl_timeout", 0.25);

        node.declare_parameter<bool>("pressure.enable", false);
        node.declare_parameter<std::string>("pressure.topic", "/auv/pressure/scaled2");
        node.declare_parameter<std::vector<double>>("pressure.extrinsic_T", {-0.24219, -0.03954, 0.01898});
        node.declare_parameter<double>("pressure.pressure_timeout", 0.25);

        node.declare_parameter<double>("dvl.velocity_cov", 4e-4);
        node.declare_parameter<double>("dvl.innovation_gate_sigma", 5.0);
        node.declare_parameter<double>("dvl.bias_init_cov", 1e-8);
        node.declare_parameter<double>("pressure.pressure_cov", 1e4);
        node.declare_parameter<double>("pressure.bias_init_cov", 1e4);
        node.declare_parameter<double>("pressure.innovation_gate_sigma", 0.0);
        node.declare_parameter<double>("pressure.fluid_density", 1025.0);
        node.declare_parameter<double>("pressure.gravity", 9.80665);
        node.declare_parameter<double>("pressure.surface_pressure", 101325.0);
        node.declare_parameter<double>("pressure.surface_z", 0.0);

        node.declare_parameter<bool>("magnetometer.enable", false);
        node.declare_parameter<std::string>("magnetometer.topic", "/auv/imu/magnetic_field");
        node.declare_parameter<std::vector<double>>("magnetometer.extrinsic_R",
            {1., 0., 0.,  0., 1., 0.,  0., 0., 1.});
        node.declare_parameter<std::vector<double>>("magnetometer.hard_iron_offset", {0.0, 0.0, 0.0});
        node.declare_parameter<std::vector<double>>("magnetometer.soft_iron_matrix",
            {1., 0., 0.,  0., 1., 0.,  0., 0., 1.});
        node.declare_parameter<double>("magnetometer.mag_cov", 1849.0);
        node.declare_parameter<double>("magnetometer.heading_cov_floor", 1e-6);
        node.declare_parameter<double>("magnetometer.innovation_gate_sigma", 3.0);
        node.declare_parameter<double>("magnetometer.mag_timeout", 0.5);
        node.declare_parameter<std::string>("magnetometer.debug_csv_path", "");
    }

    void load_parameters(rclcpp::Node &node)
    {
        node.get_parameter_or<bool>("dvl.enable", dvl_enable_, false);
        node.get_parameter_or<std::string>("dvl.topic", dvl_topic_, "/auv/dvl");
        node.get_parameter_or<double>("dvl.dvl_timeout", dvl_timeout_, 0.25);
        node.get_parameter_or<double>("dvl.velocity_cov", dvl_velocity_cov_, 4e-4);
        node.get_parameter_or<double>("dvl.innovation_gate_sigma", dvl_innovation_gate_sigma_, 5.0);
        node.get_parameter_or<double>("dvl.bias_init_cov", dvl_b_init_cov_, 1e-8);

        node.get_parameter_or<bool>("pressure.enable", pressure_enable_, false);
        node.get_parameter_or<std::string>("pressure.topic", pressure_topic_, "/auv/pressure/scaled2");
        node.get_parameter_or<double>("pressure.pressure_timeout", pressure_timeout_, 0.25);
        node.get_parameter_or<double>("pressure.pressure_cov", pressure_cov_, 1e4);
        node.get_parameter_or<double>("pressure.bias_init_cov", pressure_b_init_cov_, 1e4);
        node.get_parameter_or<double>("pressure.innovation_gate_sigma", pressure_innovation_gate_sigma_, 0.0);
        node.get_parameter_or<double>("pressure.fluid_density", pressure_fluid_density_, 1025.0);
        node.get_parameter_or<double>("pressure.gravity", pressure_gravity_, 9.80665);
        node.get_parameter_or<double>("pressure.surface_pressure", pressure_surface_pressure_, 101325.0);
        node.get_parameter_or<double>("pressure.surface_z", pressure_surface_z_, 0.0);

        std::vector<double> dvl_T;
        std::vector<double> dvl_R;
        std::vector<double> pressure_T;
        node.get_parameter_or<std::vector<double>>("dvl.extrinsic_T", dvl_T, {-0.079, -0.09691, -0.25938});
        node.get_parameter_or<std::vector<double>>("dvl.extrinsic_R", dvl_R,
                                                   {1.0, 0.0, 0.0,
                                                    0.0, 1.0, 0.0,
                                                    0.0, 0.0, 1.0});
        node.get_parameter_or<std::vector<double>>("pressure.extrinsic_T", pressure_T, {-0.24219, -0.03954, 0.01898});

        if (dvl_T.size() == 3)
        {
            dvl_T_ << dvl_T[0], dvl_T[1], dvl_T[2];
        }
        else
        {
            RCLCPP_WARN(node.get_logger(), "dvl.extrinsic_T must have 3 values. Using zero translation.");
            dvl_T_.setZero();
        }

        if (dvl_R.size() == 9)
        {
            dvl_R_ << dvl_R[0], dvl_R[1], dvl_R[2],
                      dvl_R[3], dvl_R[4], dvl_R[5],
                      dvl_R[6], dvl_R[7], dvl_R[8];
        }
        else
        {
            RCLCPP_WARN(node.get_logger(), "dvl.extrinsic_R must have 9 values. Using identity rotation.");
            dvl_R_.setIdentity();
        }

        if (pressure_T.size() == 3)
        {
            pressure_T_ << pressure_T[0], pressure_T[1], pressure_T[2];
        }
        else
        {
            RCLCPP_WARN(node.get_logger(), "pressure.extrinsic_T must have 3 values. Using zero translation.");
            pressure_T_.setZero();
        }

        node.get_parameter_or<bool>("magnetometer.enable", mag_enable_, false);
        node.get_parameter_or<std::string>("magnetometer.topic", mag_topic_, "/auv/imu/magnetic_field");
        node.get_parameter_or<double>("magnetometer.mag_cov", mag_cov_, 1849.0);
        node.get_parameter_or<double>("magnetometer.heading_cov_floor", mag_heading_cov_floor_, 1e-6);
        node.get_parameter_or<double>("magnetometer.innovation_gate_sigma", mag_innovation_gate_sigma_, 3.0);
        node.get_parameter_or<double>("magnetometer.mag_timeout", mag_timeout_, 0.5);
        node.get_parameter_or<std::string>("magnetometer.debug_csv_path", mag_debug_csv_path_, "");

        std::vector<double> mag_extrinsic_R, hard_iron, soft_iron;
        node.get_parameter_or<std::vector<double>>("magnetometer.extrinsic_R", mag_extrinsic_R,
            {1., 0., 0.,  0., 1., 0.,  0., 0., 1.});
        node.get_parameter_or<std::vector<double>>("magnetometer.hard_iron_offset", hard_iron, {0., 0., 0.});
        node.get_parameter_or<std::vector<double>>("magnetometer.soft_iron_matrix", soft_iron,
            {1., 0., 0.,  0., 1., 0.,  0., 0., 1.});

        if (mag_extrinsic_R.size() == 9)
        {
            mag_R_BM_ << mag_extrinsic_R[0], mag_extrinsic_R[1], mag_extrinsic_R[2],
                         mag_extrinsic_R[3], mag_extrinsic_R[4], mag_extrinsic_R[5],
                         mag_extrinsic_R[6], mag_extrinsic_R[7], mag_extrinsic_R[8];
        }
        else
        {
            RCLCPP_WARN(node.get_logger(),
                        "magnetometer.extrinsic_R must have 9 values. Using identity rotation.");
            mag_R_BM_.setIdentity();
        }

        if (hard_iron.size() == 3)
            mag_hard_iron_ << hard_iron[0], hard_iron[1], hard_iron[2];
        else
            mag_hard_iron_.setZero();

        if (soft_iron.size() == 9)
        {
            mag_soft_iron_ << soft_iron[0], soft_iron[1], soft_iron[2],
                              soft_iron[3], soft_iron[4], soft_iron[5],
                              soft_iron[6], soft_iron[7], soft_iron[8];
        }
        else
        {
            mag_soft_iron_.setIdentity();
        }

        dvl_timeout_ = std::max(0.0, dvl_timeout_);
        pressure_timeout_ = std::max(0.0, pressure_timeout_);
        mag_timeout_ = std::max(0.0, mag_timeout_);
        dvl_velocity_cov_ = std::max(1e-12, dvl_velocity_cov_);
        dvl_innovation_gate_sigma_ = std::max(0.0, dvl_innovation_gate_sigma_);
        dvl_b_init_cov_ = std::max(1e-12, dvl_b_init_cov_);
        pressure_cov_ = std::max(1e-6, pressure_cov_);
        pressure_b_init_cov_ = std::max(1e-12, pressure_b_init_cov_);
        pressure_innovation_gate_sigma_ = std::max(0.0, pressure_innovation_gate_sigma_);
        pressure_fluid_density_ = std::max(1e-6, pressure_fluid_density_);
        pressure_gravity_ = std::max(1e-6, pressure_gravity_);
        // Magnetometer covariance may be expressed in Tesla^2 in simulation
        // (around 1e-15) or in uT^2 for real bags. Keep only a numerical floor
        // here; a 1e-6 floor silently disables Tesla-scale magnetometer fusion.
        mag_cov_ = std::max(1e-18, mag_cov_);
        mag_heading_cov_floor_ = std::max(1e-12, mag_heading_cov_floor_);
        mag_innovation_gate_sigma_ = std::max(0.0, mag_innovation_gate_sigma_);
    }

    void create_subscriptions(rclcpp::Node &node,
                              const rclcpp::CallbackGroup::SharedPtr &callback_group = nullptr)
    {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(200000));
        qos.best_effort();
        rclcpp::SubscriptionOptions options;
        options.callback_group = callback_group;

        if (dvl_enable_)
        {
            sub_dvl_ = node.create_subscription<DvlMsg>(
                dvl_topic_, qos, std::bind(&AuxiliarySensorFusion::dvl_callback, this, std::placeholders::_1),
                options);
        }
        if (pressure_enable_)
        {
            sub_pressure_ = node.create_subscription<PressureMsg>(
                pressure_topic_, qos, std::bind(&AuxiliarySensorFusion::pressure_callback, this, std::placeholders::_1),
                options);
        }
        if (mag_enable_)
        {
            sub_mag_ = node.create_subscription<MagMsg>(
                mag_topic_, qos, std::bind(&AuxiliarySensorFusion::mag_callback, this, std::placeholders::_1),
                options);
        }
    }

    std::vector<TimedMeasurement> take_timed_measurements(double begin_time,
                                                          double end_time)
    {
        std::vector<TimedMeasurement> measurements;
        if (dvl_enable_)
        {
            const auto messages = take_dvl_measurements(begin_time, end_time);
            measurements.reserve(measurements.size() + messages.size());
            for (const auto &msg : messages)
            {
                measurements.push_back(
                    {get_time_sec(msg->header.stamp), MeasurementKind::Dvl, msg, nullptr, nullptr});
            }
        }
        if (pressure_enable_)
        {
            const auto messages = take_pressure_measurements(begin_time, end_time);
            measurements.reserve(measurements.size() + messages.size());
            for (const auto &msg : messages)
            {
                measurements.push_back(
                    {get_time_sec(msg->header.stamp), MeasurementKind::Pressure, nullptr, msg, nullptr});
            }
        }
        if (mag_enable_)
        {
            const auto messages = take_mag_measurements(begin_time, end_time);
            measurements.reserve(measurements.size() + messages.size());
            for (const auto &msg : messages)
            {
                measurements.push_back(
                    {get_time_sec(msg->header.stamp), MeasurementKind::Magnetometer, nullptr, nullptr, msg});
            }
        }

        std::stable_sort(measurements.begin(), measurements.end(),
                         [](const TimedMeasurement &a, const TimedMeasurement &b) {
                             return a.timestamp < b.timestamp;
                         });
        return measurements;
    }

    bool apply_timed_measurement(
        const TimedMeasurement &measurement,
        const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> &imu_msgs,
        Ekf &kf,
        UpdateSummary &summary)
    {
        const ImuAngularSample imu_sample =
            imu_angular_sample_at_time(measurement.timestamp, imu_msgs);

        switch (measurement.kind)
        {
            case MeasurementKind::Dvl:
            {
                if (!measurement.dvl)
                {
                    return false;
                }
                const auto &msg = *measurement.dvl;
                const state_ikfom state = kf.get_x();
                const DvlLinearization dvl =
                    build_dvl_linearization(make_dvl_observation(msg, imu_sample, state), state);
                summary.dvl_count++;
                if (dvl.valid)
                {
                    summary.dvl_res_norm_sum += dvl.residual.norm();
                    summary.dvl_res_norm_max =
                        std::max(summary.dvl_res_norm_max, dvl.residual.norm());
                    summary.dvl_res_sum += dvl.residual;
                    summary.dvl_res_abs_max =
                        summary.dvl_res_abs_max.cwiseMax(dvl.residual.cwiseAbs());
                    summary.dvl_meas_sum += dvl.measurement;
                    summary.dvl_pred_sum += dvl.prediction;
                    summary.dvl_body_vel_sum += dvl.dvl_origin_velocity_vehicle;
                }
                const bool accepted = apply_dvl_update(msg, imu_sample, kf);
                summary.dvl_accepted += accepted ? 1 : 0;
                summary.dvl_rejected += accepted ? 0 : 1;
                summary.dvl_updated = accepted || summary.dvl_updated;
                return accepted;
            }
            case MeasurementKind::Pressure:
            {
                if (!measurement.pressure)
                {
                    return false;
                }
                const auto &msg = *measurement.pressure;
                const double residual_pa = pressure_residual(msg, kf.get_x());
                const double residual_depth = residual_pa / pressure_scale();
                summary.pressure_count++;
                summary.pressure_res_depth_sum += std::abs(residual_depth);
                summary.pressure_res_depth_max = std::max(summary.pressure_res_depth_max,
                                                          std::abs(residual_depth));
                const bool accepted = apply_pressure_update(msg, kf);
                summary.pressure_accepted += accepted ? 1 : 0;
                summary.pressure_rejected += accepted ? 0 : 1;
                summary.pressure_updated = accepted || summary.pressure_updated;
                return accepted;
            }
            case MeasurementKind::Magnetometer:
            {
                if (!measurement.magnetometer)
                {
                    return false;
                }
                const auto &msg = *measurement.magnetometer;
                double residual = 0.0;
                const bool accepted = apply_mag_update(msg, kf, &residual);
                summary.mag_count++;
                summary.mag_res_norm_sum += std::abs(residual);
                summary.mag_res_norm_max =
                    std::max(summary.mag_res_norm_max, std::abs(residual));
                summary.mag_accepted += accepted ? 1 : 0;
                summary.mag_rejected += accepted ? 0 : 1;
                summary.mag_updated = accepted || summary.mag_updated;
                return accepted;
            }
        }
        return false;
    }

    LateMeasurementCounts take_late_measurement_counts()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const LateMeasurementCounts counts = late_measurement_counts_;
        late_measurement_counts_ = {};
        return counts;
    }

    void warn_timeouts(rclcpp::Node &node, double end_time) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (dvl_enable_ && last_timestamp_dvl_ > 0.0 && end_time - last_timestamp_dvl_ > dvl_timeout_)
        {
            RCLCPP_WARN_THROTTLE(node.get_logger(), *node.get_clock(), 5000,
                                 "No DVL message for %.2f s (timeout %.2f s).",
                                 end_time - last_timestamp_dvl_, dvl_timeout_);
        }
        if (pressure_enable_ && last_timestamp_pressure_ > 0.0 && end_time - last_timestamp_pressure_ > pressure_timeout_)
        {
            RCLCPP_WARN_THROTTLE(node.get_logger(), *node.get_clock(), 5000,
                                 "No pressure message for %.2f s (timeout %.2f s).",
                                 end_time - last_timestamp_pressure_, pressure_timeout_);
        }
        if (mag_enable_ && last_timestamp_mag_ > 0.0 && end_time - last_timestamp_mag_ > mag_timeout_)
        {
            RCLCPP_WARN_THROTTLE(node.get_logger(), *node.get_clock(), 5000,
                                 "No magnetometer message for %.2f s (timeout %.2f s).",
                                 end_time - last_timestamp_mag_, mag_timeout_);
        }
    }

    bool dvl_enabled() const { return dvl_enable_; }
    bool pressure_enabled() const { return pressure_enable_; }
    bool mag_enabled() const { return mag_enable_; }

    /// Apply an externally-built linear measurement through the SAME Joseph-form update and
    /// post-boxplus covariance transport the DVL/pressure/magnetometer updates use.
    ///
    /// Exposed so a measurement source outside this class (the monocular visual update) can be
    /// a pure producer of (residual, H, R) rather than duplicating the filter mathematics --
    /// duplicating it is how the two implementations drift apart.
    bool apply_external_update(const Eigen::VectorXd &residual,
                               const Eigen::MatrixXd &H,
                               const Eigen::MatrixXd &R,
                               Ekf &kf)
    {
        return apply_linear_update(residual, H, R, kf);
    }
    const std::string &dvl_topic() const { return dvl_topic_; }
    const std::string &pressure_topic() const { return pressure_topic_; }
    const std::string &mag_topic() const { return mag_topic_; }
    double dvl_timeout() const { return dvl_timeout_; }
    double pressure_timeout() const { return pressure_timeout_; }
    double mag_timeout() const { return mag_timeout_; }
    const V3D &dvl_T() const { return dvl_T_; }
    const V3D &pressure_T() const { return pressure_T_; }
    double dvl_velocity_cov() const { return dvl_velocity_cov_; }
    double dvl_innovation_gate_sigma() const { return dvl_innovation_gate_sigma_; }
    double dvl_b_init_cov() const { return dvl_b_init_cov_; }
    double pressure_b_init_cov() const { return pressure_b_init_cov_; }

    // Pressure measures depth along the world vertical axis, while the FAST-LIO
    // state is expressed in camera_init. Store the static camera_init pose so the
    // pressure model can project local sensor position onto world z.
    void set_camera_init_pose_in_world(const V3D &t, const M3D &R)
    {
        camera_init_T_in_world_ = t;
        camera_init_R_in_world_ = R;
    }

    void initialize_pressure_reference_pose(const state_ikfom &state)
    {
        if (!pressure_enable_)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pressure_reference_pose_ready_)
        {
            pressure_reference_sensor_z_world_ = pressure_sensor_z_world(state);
            pressure_reference_pose_ready_ = true;
        }
    }

private:
    struct ImuAngularSample
    {
        V3D raw_gyro = V3D::Zero();
        M3D covariance = M3D::Zero();
        bool covariance_valid = false;
        bool valid = false;
    };

    struct DvlObservation
    {
        V3D measurement = V3D::Zero();
        M3D covariance = M3D::Identity();
        ImuAngularSample imu;
        bool valid = false;
    };

    struct DvlLinearization
    {
        V3D measurement = V3D::Zero();
        V3D prediction = V3D::Zero();
        V3D residual = V3D::Zero();
        V3D dvl_origin_velocity_vehicle = V3D::Zero();
        Eigen::Matrix<double, 3, state_ikfom::DOF> H =
            Eigen::Matrix<double, 3, state_ikfom::DOF>::Zero();
        M3D R = M3D::Identity();
        bool valid = false;
    };

    static bool finite_vector(const Eigen::VectorXd &v)
    {
        return v.allFinite();
    }

    double pressure_scale() const
    {
        return pressure_fluid_density_ * pressure_gravity_;
    }

    // Use the message covariance when valid, but never below the configured fallback.
    // This prevents unrealistically tight bag-recorded covariances from overwhelming the EKF.
    double covariance_or_fallback(double value, double fallback) const
    {
        if (!std::isfinite(value) || value <= 1e-12)
        {
            return fallback;
        }
        return std::max(value, fallback);
    }

    ImuAngularSample imu_angular_sample(const sensor_msgs::msg::Imu &msg) const
    {
        ImuAngularSample sample;
        const auto &gyro = msg.angular_velocity;
        sample.raw_gyro = V3D(gyro.x, gyro.y, gyro.z);
        sample.valid = sample.raw_gyro.allFinite();

        if (msg.angular_velocity_covariance[0] < 0.0)
        {
            return sample;
        }
        M3D covariance;
        covariance << msg.angular_velocity_covariance[0],
                      msg.angular_velocity_covariance[1],
                      msg.angular_velocity_covariance[2],
                      msg.angular_velocity_covariance[3],
                      msg.angular_velocity_covariance[4],
                      msg.angular_velocity_covariance[5],
                      msg.angular_velocity_covariance[6],
                      msg.angular_velocity_covariance[7],
                      msg.angular_velocity_covariance[8];
        covariance = 0.5 * (covariance + covariance.transpose()).eval();
        if (!covariance.allFinite() || covariance.cwiseAbs().maxCoeff() <= 1e-18)
        {
            return sample;
        }
        Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
        if (solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() >= -1e-12)
        {
            sample.covariance = covariance;
            sample.covariance_valid = true;
        }
        return sample;
    }

    // Pick the raw IMU sample whose stamp is closest to the DVL timestamp.
    ImuAngularSample imu_angular_sample_at_time(
        double target_t,
        const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> &imu_msgs) const
    {
        if (imu_msgs.empty()) return ImuAngularSample{};
        auto best = imu_msgs.begin();
        double best_diff = std::abs(get_time_sec((*best)->header.stamp) - target_t);
        for (auto it = imu_msgs.begin(); it != imu_msgs.end(); ++it)
        {
            const double d = std::abs(get_time_sec((*it)->header.stamp) - target_t);
            if (d < best_diff) { best_diff = d; best = it; }
        }
        return imu_angular_sample(**best);
    }

    V3D dvl_measurement(const DvlMsg &msg) const
    {
        const auto &linear = msg.twist.twist.linear;
        return V3D(linear.x, linear.y, linear.z);
    }

    M3D dvl_measurement_covariance(const DvlMsg &msg) const
    {
        M3D covariance;
        covariance << msg.twist.covariance[0], msg.twist.covariance[1], msg.twist.covariance[2],
                      msg.twist.covariance[6], msg.twist.covariance[7], msg.twist.covariance[8],
                      msg.twist.covariance[12], msg.twist.covariance[13], msg.twist.covariance[14];
        covariance = 0.5 * (covariance + covariance.transpose()).eval();

        bool use_message_covariance = covariance.allFinite();
        if (use_message_covariance)
        {
            Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
            use_message_covariance =
                solver.info() == Eigen::Success &&
                solver.eigenvalues().minCoeff() >= -1e-12 &&
                covariance.diagonal().minCoeff() > 1e-12;
        }
        if (!use_message_covariance)
        {
            covariance = M3D::Identity() * dvl_velocity_cov_;
        }
        else
        {
            for (int i = 0; i < 3; ++i)
            {
                covariance(i, i) = std::max(covariance(i, i), dvl_velocity_cov_);
            }
        }
        return covariance;
    }

    DvlObservation make_dvl_observation(const DvlMsg &msg,
                                        const ImuAngularSample &imu_sample,
                                        const state_ikfom &state) const
    {
        DvlObservation observation;
        observation.measurement = dvl_measurement(msg);
        observation.covariance = dvl_measurement_covariance(msg);
        observation.imu = imu_sample;
        observation.valid = observation.measurement.allFinite() &&
                            observation.covariance.allFinite() &&
                            observation.imu.valid;
        return observation;
    }

    DvlLinearization build_dvl_linearization(const DvlObservation &observation,
                                             const state_ikfom &state) const
    {
        DvlLinearization result;
        if (!observation.valid)
        {
            return result;
        }

        const auto model = underwater_fastlio::dvl::evaluate(
            state, observation.imu.raw_gyro, dvl_R_, dvl_T_);
        result.measurement = observation.measurement;
        result.prediction = model.prediction_dvl;
        result.residual = result.measurement - result.prediction;
        result.dvl_origin_velocity_vehicle = model.dvl_origin_velocity_vehicle;
        result.H = model.H;
        result.R = observation.covariance;

        if (observation.imu.covariance_valid && dvl_T_.squaredNorm() > 0.0)
        {
            const M3D C_DV = dvl_R_.transpose();
            const M3D p_cross = underwater_fastlio::dvl::skew(dvl_T_);
            result.R += C_DV * p_cross * observation.imu.covariance *
                        p_cross.transpose() * C_DV.transpose();
        }
        result.R = 0.5 * (result.R + result.R.transpose()).eval();
        Eigen::LLT<M3D> llt(result.R);
        result.valid = result.measurement.allFinite() &&
                       result.prediction.allFinite() &&
                       result.residual.allFinite() &&
                       result.H.allFinite() &&
                       result.R.allFinite() &&
                       llt.info() == Eigen::Success;
        return result;
    }

    bool dvl_passes_gate(const DvlLinearization &dvl,
                         const typename Ekf::cov &P) const
    {
        if (!dvl.valid)
        {
            return false;
        }
        if (dvl_innovation_gate_sigma_ <= 0.0)
        {
            return true;
        }

        const M3D innovation_covariance = dvl.H * P * dvl.H.transpose() + dvl.R;
        Eigen::LDLT<M3D> ldlt(innovation_covariance);
        if (ldlt.info() != Eigen::Success)
        {
            return false;
        }
        const double nis = dvl.residual.dot(ldlt.solve(dvl.residual));
        const double threshold = 3.0 * dvl_innovation_gate_sigma_ *
                                 dvl_innovation_gate_sigma_;
        return std::isfinite(nis) && nis <= threshold;
    }

    Eigen::RowVector3d world_z_axis_in_camera_init() const
    {
        return camera_init_R_in_world_.row(2);
    }

    double pressure_sensor_z_world(const state_ikfom &state) const
    {
        V3D pressure_offset_ci = V3D::Zero();
        pressure_offset_ci.z() = pressure_T_.z();
        const V3D sensor_ci = state.pos + pressure_offset_ci;
        return camera_init_T_in_world_.z() + world_z_axis_in_camera_init().dot(sensor_ci);
    }

    double pressure_raw_depth(const state_ikfom &state) const
    {
        return pressure_surface_z_ - pressure_sensor_z_world(state);
    }

    double pressure_prediction(const state_ikfom &state) const
    {
        const double relative_depth = pressure_reference_sensor_z_world_ - pressure_sensor_z_world(state);
        return pressure_surface_pressure_ + pressure_scale() * relative_depth + state.b_pressure[0];
    }

    double pressure_residual(const PressureMsg &msg, const state_ikfom &state) const
    {
        return msg.fluid_pressure - pressure_prediction(state);
    }

    bool pressure_reference_ready_or_collect(const PressureMsg &msg,
                                             const state_ikfom &state)
    {
        if (pressure_ref_finalized_)
        {
            return true;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (pressure_ref_finalized_)
        {
            return true;
        }
        if (!pressure_reference_pose_ready_)
        {
            return false;
        }

        // The state has been propagated to this pressure timestamp. Express
        // every startup sample at the one stored reference pose before taking
        // the mean, so vehicle motion during initialization cannot create an
        // artificial depth offset.
        const double relative_depth =
            pressure_reference_sensor_z_world_ - pressure_sensor_z_world(state);
        pressure_init_sum_ +=
            msg.fluid_pressure - pressure_scale() * relative_depth;
        ++pressure_init_samples_collected_;
        if (pressure_init_samples_collected_ < kPressureReferenceSamples)
        {
            return false;
        }

        const double mean_pressure =
            pressure_init_sum_ / static_cast<double>(pressure_init_samples_collected_);
        pressure_surface_pressure_ = mean_pressure - state.b_pressure[0];
        pressure_ref_finalized_ = true;
        // Reference samples calibrate the local datum and are not reused as
        // independent Kalman measurements. Fusion starts with the next sample.
        return false;
    }

    bool apply_dvl_update(const DvlMsg &msg,
                          const ImuAngularSample &imu_sample,
                          Ekf &kf)
    {
        // Until the startup-relative magnetic heading is initialized, yaw is
        // unobservable in a stand-alone body-frame velocity update. Deferring
        // these first few DVL samples prevents an arbitrary startup yaw while
        // retaining the complete DVL Jacobian and gain thereafter.
        if (mag_enable_ && !mag_reference_is_ready())
        {
            return false;
        }
        const state_ikfom state = kf.get_x();
        const DvlLinearization dvl =
            build_dvl_linearization(make_dvl_observation(msg, imu_sample, state), state);
        if (!dvl_passes_gate(dvl, kf.get_P()))
        {
            return false;
        }
        return apply_linear_update(dvl.residual, dvl.H, dvl.R, kf);
    }

    bool apply_pressure_update(const PressureMsg &msg, Ekf &kf)
    {
        const state_ikfom state = kf.get_x();
        if (!pressure_reference_ready_or_collect(msg, state))
        {
            return false;
        }

        Eigen::VectorXd residual(1);
        residual(0) = pressure_residual(msg, state);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, state_ikfom::DOF);
        // Pressure measures World-z depth, but it must not correct horizontal
        // x/y. Use the World-z prediction and let the EKF correct only local depth.
        H(0, 2) = -pressure_scale() * world_z_axis_in_camera_init().z();
        H(0, 26) = 1.0;

        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, 1);
        R(0, 0) = covariance_or_fallback(msg.variance, pressure_cov_);

        if (pressure_innovation_gate_sigma_ > 0.0)
        {
            const typename Ekf::cov P = kf.get_P();
            const Eigen::MatrixXd S = H * P * H.transpose() + R;
            if (!S.allFinite() || S(0, 0) <= 0.0)
            {
                return false;
            }
            const double sigma_pres = std::sqrt(S(0, 0));
            if (std::abs(residual(0)) > pressure_innovation_gate_sigma_ * sigma_pres)
            {
                return false;
            }
        }

        return apply_pressure_depth_update(residual, H, R, kf);
    }

    bool apply_pressure_depth_update(const Eigen::VectorXd &residual,
                                     const Eigen::MatrixXd &H,
                                     const Eigen::MatrixXd &R,
                                     Ekf &kf)
    {
        if (residual.size() != 1 || H.rows() != 1 || H.cols() != state_ikfom::DOF ||
            R.rows() != 1 || R.cols() != 1 || !finite_vector(residual) ||
            !H.allFinite() || !R.allFinite())
        {
            return false;
        }

        typename Ekf::cov P = kf.get_P();
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
        if (ldlt.info() != Eigen::Success)
        {
            return false;
        }

        Eigen::MatrixXd K = P * H.transpose() * ldlt.solve(Eigen::MatrixXd::Identity(1, 1));
        for (int row = 0; row < K.rows(); ++row)
        {
            if (row != 2 && row != 26)
            {
                K.row(row).setZero();
            }
        }

        const Eigen::VectorXd dx_dyn = K * residual;
        if (!finite_vector(dx_dyn))
        {
            return false;
        }

        typename Ekf::vectorized_state dx = Ekf::vectorized_state::Zero();
        dx = dx_dyn;

        state_ikfom updated_state = kf.get_x();
        updated_state.boxplus(dx);

        const typename Ekf::cov I_state = Ekf::cov::Identity();
        typename Ekf::cov KH = K * H;
        typename Ekf::cov P_new = (I_state - KH) * P * (I_state - KH).transpose() + K * R * K.transpose();
        P_new = 0.5 * (P_new + P_new.transpose()).eval();

        kf.change_x(updated_state);
        kf.change_P(P_new);
        return true;
    }

    typename Ekf::cov transport_covariance_after_boxplus(
        const typename Ekf::cov &P,
        state_ikfom prior_state,
        state_ikfom updated_state,
        const typename Ekf::vectorized_state &dx) const
    {
        if (prior_state.SO3_state.empty() || prior_state.S2_state.empty())
        {
            prior_state.build_SO3_state();
            prior_state.build_S2_state();
        }
        if (updated_state.SO3_state.empty() || updated_state.S2_state.empty())
        {
            updated_state.build_SO3_state();
            updated_state.build_S2_state();
        }

        typename Ekf::cov transport = Ekf::cov::Identity();
        for (const auto &entry : updated_state.SO3_state)
        {
            const int index = entry.first;
            MTK::vect<3, double> segment;
            segment << dx(index), dx(index + 1), dx(index + 2);
            transport.template block<3, 3>(index, index) =
                MTK::A_matrix(segment).transpose();
        }
        for (const auto &entry : updated_state.S2_state)
        {
            const int index = entry.first;
            MTK::vect<2, double> segment;
            segment << dx(index), dx(index + 1);
            Eigen::Matrix<double, 2, 3> Nx;
            Eigen::Matrix<double, 3, 2> Mx;
            updated_state.S2_Nx_yy(Nx, index);
            prior_state.S2_Mx(Mx, segment, index);
            transport.template block<2, 2>(index, index) = Nx * Mx;
        }

        typename Ekf::cov transported = transport * P * transport.transpose();
        return 0.5 * (transported + transported.transpose()).eval();
    }

    bool apply_linear_update(const Eigen::VectorXd &residual,
                             const Eigen::MatrixXd &H,
                             const Eigen::MatrixXd &R,
                             Ekf &kf)
    {
        if (residual.size() == 0 || H.rows() != residual.size() || H.cols() != state_ikfom::DOF ||
            R.rows() != residual.size() || R.cols() != residual.size() || !finite_vector(residual) ||
            !H.allFinite() || !R.allFinite())
        {
            return false;
        }

        typename Ekf::cov P = kf.get_P();
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
        if (ldlt.info() != Eigen::Success)
        {
            return false;
        }

        const Eigen::MatrixXd I_meas = Eigen::MatrixXd::Identity(residual.size(), residual.size());
        Eigen::MatrixXd K = P * H.transpose() * ldlt.solve(I_meas);
        const Eigen::VectorXd dx_dyn = K * residual;
        if (!finite_vector(dx_dyn))
        {
            return false;
        }

        typename Ekf::vectorized_state dx = Ekf::vectorized_state::Zero();
        dx = dx_dyn;

        const state_ikfom prior_state = kf.get_x();
        state_ikfom updated_state = prior_state;
        updated_state.boxplus(dx);

        const typename Ekf::cov I_state = Ekf::cov::Identity();
        typename Ekf::cov KH = K * H;
        typename Ekf::cov P_new = (I_state - KH) * P * (I_state - KH).transpose() + K * R * K.transpose();
        P_new = 0.5 * (P_new + P_new.transpose()).eval();
        P_new = transport_covariance_after_boxplus(P_new, prior_state, updated_state, dx);

        kf.change_x(updated_state);
        kf.change_P(P_new);
        return true;
    }

    std::vector<DvlMsg::ConstSharedPtr> take_dvl_measurements(double begin_time, double end_time)
    {
        std::vector<DvlMsg::ConstSharedPtr> messages;
        std::lock_guard<std::mutex> lock(mutex_);
        while (!dvl_buffer_.empty())
        {
            const double stamp = get_time_sec(dvl_buffer_.front()->header.stamp);
            if (stamp <= begin_time + 1e-6)
            {
                ++late_measurement_counts_.dvl;
                dvl_buffer_.pop_front();
                continue;
            }
            if (stamp > end_time + 1e-6)
            {
                break;
            }
            messages.push_back(dvl_buffer_.front());
            dvl_buffer_.pop_front();
        }
        return messages;
    }

    std::vector<PressureMsg::ConstSharedPtr> take_pressure_measurements(double begin_time, double end_time)
    {
        std::vector<PressureMsg::ConstSharedPtr> messages;
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pressure_buffer_.empty())
        {
            const double stamp = get_time_sec(pressure_buffer_.front()->header.stamp);
            if (stamp <= begin_time + 1e-6)
            {
                ++late_measurement_counts_.pressure;
                pressure_buffer_.pop_front();
                continue;
            }
            if (stamp > end_time + 1e-6)
            {
                break;
            }
            messages.push_back(pressure_buffer_.front());
            pressure_buffer_.pop_front();
        }
        return messages;
    }

    void dvl_callback(const DvlMsg::ConstSharedPtr msg)
    {
        const double timestamp = get_time_sec(msg->header.stamp);
        std::lock_guard<std::mutex> lock(mutex_);
        if (last_timestamp_dvl_ >= 0.0 &&
            std::abs(timestamp - last_timestamp_dvl_) <= 1e-9)
        {
            return;
        }
        if (timestamp < last_timestamp_dvl_)
        {
            dvl_buffer_.clear();
        }
        last_timestamp_dvl_ = timestamp;
        dvl_buffer_.push_back(msg);
    }

    void pressure_callback(const PressureMsg::ConstSharedPtr msg)
    {
        const double timestamp = get_time_sec(msg->header.stamp);
        std::lock_guard<std::mutex> lock(mutex_);
        if (last_timestamp_pressure_ >= 0.0 &&
            std::abs(timestamp - last_timestamp_pressure_) <= 1e-9)
        {
            return;
        }
        if (timestamp < last_timestamp_pressure_)
        {
            pressure_buffer_.clear();
            pressure_init_sum_ = 0.0;
            pressure_init_samples_collected_ = 0;
            pressure_ref_finalized_ = false;
            pressure_reference_pose_ready_ = false;
        }
        last_timestamp_pressure_ = timestamp;
        pressure_buffer_.push_back(msg);
    }

    void mag_callback(const MagMsg::ConstSharedPtr msg)
    {
        const double timestamp = get_time_sec(msg->header.stamp);
        std::lock_guard<std::mutex> lock(mutex_);
        if (last_timestamp_mag_ >= 0.0 &&
            std::abs(timestamp - last_timestamp_mag_) <= 1e-9)
        {
            return;
        }
        if (timestamp < last_timestamp_mag_)
        {
            mag_buffer_.clear();
            reset_mag_reference_locked();
        }
        last_timestamp_mag_ = timestamp;
        mag_buffer_.push_back(msg);
    }

    // R_BM rotates the calibrated magnetometer-frame field into the IMU/body frame.
    V3D mag_corrected(const MagMsg &msg) const
    {
        const V3D raw(msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z);
        return mag_R_BM_ * mag_soft_iron_ * (raw - mag_hard_iron_);
    }

    M3D mag_calibrated_covariance() const
    {
        const M3D calibration = mag_R_BM_ * mag_soft_iron_;
        return mag_cov_ * calibration * calibration.transpose();
    }

    void reset_mag_reference_locked()
    {
        mag_reference_sum_local_.setZero();
        mag_reference_samples_local_.clear();
        mag_reference_sample_count_ = 0;
        mag_reference_ready_ = false;
        mag_vertical_local_ = V3D::UnitZ();
        mag_reference_local_.setZero();
        mag_horizontal_reference_local_.setZero();
        mag_reference_heading_variance_ = 0.0;
    }

    bool mag_reference_is_ready() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return mag_reference_ready_;
    }

    bool collect_mag_reference_sample(const state_ikfom &state,
                                      const V3D &magnetic_body,
                                      const char *&reason)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (mag_reference_ready_)
        {
            return true;
        }

        if (mag_reference_sample_count_ == 0)
        {
            const V3D gravity_local(state.grav[0], state.grav[1], state.grav[2]);
            mag_vertical_local_ = V3D::UnitZ();
            if (gravity_local.norm() > 1e-6)
            {
                mag_vertical_local_ = (-gravity_local).normalized();
            }
        }

        if (!magnetic_body.allFinite() || magnetic_body.norm() <= 1e-12)
        {
            reason = "invalid_sample";
            return false;
        }

        const V3D sample_local = state.rot.toRotationMatrix() * magnetic_body;
        V3D running_mean = V3D::Zero();
        if (mag_reference_sample_count_ > 0)
        {
            running_mean =
                mag_reference_sum_local_ / static_cast<double>(mag_reference_sample_count_);
        }
        if (!underwater_fastlio::magnetometer::reference_sample_is_inlier(
                sample_local, running_mean, mag_reference_sample_count_))
        {
            reason = "reference_outlier";
            return false;
        }

        mag_reference_sum_local_ += sample_local;
        mag_reference_samples_local_.push_back(sample_local);
        ++mag_reference_sample_count_;
        if (mag_reference_sample_count_ < kMagReferenceSamples)
        {
            reason = "reference_initializing";
            return false;
        }

        mag_reference_local_ =
            mag_reference_sum_local_ / static_cast<double>(mag_reference_sample_count_);
        const V3D n = mag_vertical_local_;
        const V3D horizontal =
            mag_reference_local_ - n * n.dot(mag_reference_local_);
        if (mag_reference_local_.norm() <= 1e-12 || horizontal.norm() <= 1e-12)
        {
            reset_mag_reference_locked();
            reason = "reference_horizontal_field_too_small";
            return false;
        }

        mag_horizontal_reference_local_ = horizontal.normalized();
        double heading_sq_sum = 0.0;
        for (const V3D &sample : mag_reference_samples_local_)
        {
            V3D sample_horizontal = sample - n * n.dot(sample);
            if (sample_horizontal.norm() <= 1e-12)
            {
                continue;
            }
            sample_horizontal.normalize();
            const double angle = std::atan2(
                n.dot(mag_horizontal_reference_local_.cross(sample_horizontal)),
                mag_horizontal_reference_local_.dot(sample_horizontal));
            heading_sq_sum += angle * angle;
        }
        mag_reference_heading_variance_ =
            heading_sq_sum /
            (static_cast<double>(mag_reference_sample_count_) *
             static_cast<double>(mag_reference_sample_count_));
        mag_reference_ready_ = true;
        reason = "reference_ready";
        return false;
    }

    std::vector<MagMsg::ConstSharedPtr> take_mag_measurements(double begin_time, double end_time)
    {
        std::vector<MagMsg::ConstSharedPtr> messages;
        std::lock_guard<std::mutex> lock(mutex_);
        while (!mag_buffer_.empty())
        {
            const double stamp = get_time_sec(mag_buffer_.front()->header.stamp);
            if (stamp <= begin_time + 1e-6)
            {
                ++late_measurement_counts_.magnetometer;
                mag_buffer_.pop_front();
                continue;
            }
            if (stamp > end_time + 1e-6)
                break;
            messages.push_back(mag_buffer_.front());
            mag_buffer_.pop_front();
        }
        return messages;
    }

    bool apply_mag_update(const MagMsg &msg,
                          Ekf &kf,
                          double *innovation_out = nullptr)
    {
        if (innovation_out)
        {
            *innovation_out = 0.0;
        }

        const state_ikfom state = kf.get_x();
        const V3D measured = mag_corrected(msg);
        const double yaw_before = yaw_from_state(state);

        const char *reference_reason = "reference_initializing";
        if (!collect_mag_reference_sample(state, measured, reference_reason))
        {
            write_mag_debug_row(msg, measured, {}, 0.0, 0.0, 0.0,
                                V3D::Zero(), V3D::Zero(), V3D::Zero(),
                                false, reference_reason, yaw_before, yaw_before,
                                0.0, 0.0, 0.0);
            return false;
        }
        V3D h0;
        V3D vertical_local;
        double reference_variance = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            h0 = mag_horizontal_reference_local_;
            vertical_local = mag_vertical_local_;
            reference_variance = mag_reference_heading_variance_;
        }

        const auto observation = underwater_fastlio::magnetometer::evaluate(
            state.rot.toRotationMatrix(), measured, h0, vertical_local);
        if (!observation.valid)
        {
            write_mag_debug_row(msg, measured, observation, 0.0, 0.0, 0.0,
                                V3D::Zero(), V3D::Zero(), V3D::Zero(),
                                false, "heading_geometry_degenerate", yaw_before, yaw_before,
                                0.0, 0.0, 0.0);
            return false;
        }
        if (innovation_out)
        {
            *innovation_out = observation.innovation;
        }

        const M3D calibrated_covariance = mag_calibrated_covariance();
        double heading_covariance =
            (observation.magnetic_jacobian * calibrated_covariance *
             observation.magnetic_jacobian.transpose())(0, 0) +
            mag_heading_cov_floor_ + reference_variance;
        heading_covariance = std::max(1e-12, heading_covariance);

        Eigen::RowVectorXd H = Eigen::RowVectorXd::Zero(state_ikfom::DOF);
        H.segment<3>(3) = observation.observation_jacobian;
        typename Ekf::cov P = kf.get_P();
        const double innovation_variance = (H * P * H.transpose())(0, 0) + heading_covariance;
        if (!H.allFinite() || !P.allFinite() || !std::isfinite(innovation_variance) ||
            innovation_variance <= 0.0)
        {
            write_mag_debug_row(msg, measured, observation, heading_covariance,
                                innovation_variance, 0.0, V3D::Zero(), V3D::Zero(), V3D::Zero(),
                                false, "invalid_innovation_covariance", yaw_before, yaw_before,
                                0.0, 0.0, 0.0);
            return false;
        }

        const double nis = observation.innovation * observation.innovation /
                           innovation_variance;
        if (mag_innovation_gate_sigma_ > 0.0 &&
            nis > mag_innovation_gate_sigma_ * mag_innovation_gate_sigma_)
        {
            write_mag_debug_row(msg, measured, observation, heading_covariance,
                                innovation_variance, nis, V3D::Zero(), V3D::Zero(), V3D::Zero(),
                                false, "innovation_nis", yaw_before, yaw_before,
                                0.0, 0.0, 0.0);
            return false;
        }

        const Eigen::VectorXd K = underwater_fastlio::magnetometer::constrained_gain(
            P, H, innovation_variance, observation.g, 3);
        const Eigen::VectorXd dx_dyn = K * observation.innovation;
        if (!finite_vector(dx_dyn))
        {
            write_mag_debug_row(msg, measured, observation, heading_covariance,
                                innovation_variance, nis, K.segment<3>(3), K.segment<3>(15),
                                V3D::Zero(), false, "nonfinite_dx", yaw_before, yaw_before,
                                0.0, 0.0, 0.0);
            return false;
        }

        typename Ekf::vectorized_state dx = Ekf::vectorized_state::Zero();
        dx = dx_dyn;
        state_ikfom updated_state = state;
        updated_state.boxplus(dx);

        const V3D g_after =
            updated_state.rot.toRotationMatrix().transpose() * vertical_local;
        const double tilt_axis_change = (g_after - observation.g).norm();
        Eigen::VectorXd protected_dx = dx_dyn;
        protected_dx.segment<3>(3).setZero();
        protected_dx.segment<3>(15).setZero();
        const double protected_state_change = protected_dx.cwiseAbs().maxCoeff();
        if (tilt_axis_change > 1e-10 || protected_state_change > 1e-14)
        {
            write_mag_debug_row(msg, measured, observation, heading_covariance,
                                innovation_variance, nis, K.segment<3>(3), K.segment<3>(15),
                                dx_dyn.segment<3>(3), false, "roll_pitch_protection_failed",
                                yaw_before, yaw_before, tilt_axis_change,
                                protected_state_change, 0.0);
            return false;
        }

        Eigen::MatrixXd P_new = underwater_fastlio::magnetometer::joseph_covariance(
            P, H, K, heading_covariance);
        P_new = underwater_fastlio::magnetometer::transport_attitude_covariance(
            P_new, dx_dyn.segment<3>(3), 3);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covariance_solver(P_new);
        const double min_eigenvalue = covariance_solver.info() == Eigen::Success
            ? covariance_solver.eigenvalues().minCoeff()
            : -std::numeric_limits<double>::infinity();
        if (!underwater_fastlio::magnetometer::covariance_is_psd(P_new, 1e-9))
        {
            write_mag_debug_row(msg, measured, observation, heading_covariance,
                                innovation_variance, nis, K.segment<3>(3), K.segment<3>(15),
                                dx_dyn.segment<3>(3), false, "covariance_not_psd",
                                yaw_before, yaw_before, tilt_axis_change,
                                protected_state_change, min_eigenvalue);
            return false;
        }

        const auto after_observation = underwater_fastlio::magnetometer::evaluate(
            updated_state.rot.toRotationMatrix(), measured, h0, vertical_local);
        const double yaw_after = yaw_from_state(updated_state);
        kf.change_x(updated_state);
        typename Ekf::cov fixed_P = P_new;
        kf.change_P(fixed_P);
        write_mag_debug_row(msg, measured, observation, heading_covariance,
                            innovation_variance, nis, K.segment<3>(3), K.segment<3>(15),
                            dx_dyn.segment<3>(3), true, "accepted", yaw_before, yaw_after,
                            tilt_axis_change, protected_state_change, min_eigenvalue,
                            after_observation.valid ? after_observation.innovation : 0.0);
        return true;
    }

    double yaw_from_state(const state_ikfom &state) const
    {
        const M3D R = state.rot.toRotationMatrix();
        return std::atan2(R(1, 0), R(0, 0));
    }

    void write_mag_debug_row(const MagMsg &msg,
                             const V3D &measured,
                             const underwater_fastlio::magnetometer::HeadingObservation &observation,
                             double heading_covariance,
                             double innovation_cov,
                             double nis,
                             const V3D &attitude_gain,
                             const V3D &gyro_bias_gain,
                             const V3D &attitude_correction,
                             bool accepted,
                             const char *reason,
                             double yaw_before,
                             double yaw_after,
                             double tilt_axis_change,
                             double protected_state_change,
                             double min_covariance_eigenvalue,
                             double innovation_after = 0.0)
    {
        if (mag_debug_csv_path_.empty())
        {
            return;
        }
        if (!mag_debug_csv_.is_open())
        {
            mag_debug_csv_.open(mag_debug_csv_path_, std::ios::out | std::ios::trunc);
            if (!mag_debug_csv_.is_open())
            {
                mag_debug_csv_path_.clear();
                return;
            }
            mag_debug_csv_ << "stamp,accepted,reason,innovation_rad,innovation_after_rad,heading_cov,S,nis,"
                              "yaw_before_rad,yaw_after_rad,tilt_axis_change,protected_state_change,min_cov_eig,"
                              "meas_x,meas_y,meas_z,g_x,g_y,g_z,p_x,p_y,p_z,"
                              "k_theta_x,k_theta_y,k_theta_z,k_bg_x,k_bg_y,k_bg_z,"
                              "dx_theta_x,dx_theta_y,dx_theta_z\n";
        }
        mag_debug_csv_ << std::setprecision(12)
                       << get_time_sec(msg.header.stamp) << ','
                       << (accepted ? 1 : 0) << ','
                       << reason << ','
                       << observation.innovation << ','
                       << innovation_after << ','
                       << heading_covariance << ','
                       << innovation_cov << ','
                       << nis << ','
                       << yaw_before << ','
                       << yaw_after << ','
                       << tilt_axis_change << ','
                       << protected_state_change << ','
                       << min_covariance_eigenvalue << ','
                       << measured.x() << ','
                       << measured.y() << ','
                       << measured.z() << ','
                       << observation.g.x() << ','
                       << observation.g.y() << ','
                       << observation.g.z() << ','
                       << observation.p.x() << ','
                       << observation.p.y() << ','
                       << observation.p.z() << ','
                       << attitude_gain.x() << ','
                       << attitude_gain.y() << ','
                       << attitude_gain.z() << ','
                       << gyro_bias_gain.x() << ','
                       << gyro_bias_gain.y() << ','
                       << gyro_bias_gain.z() << ','
                       << attitude_correction.x() << ','
                       << attitude_correction.y() << ','
                       << attitude_correction.z() << '\n';
    }

    bool dvl_enable_ = false;
    bool pressure_enable_ = false;
    bool mag_enable_ = false;
    std::string dvl_topic_ = "/auv/dvl";
    std::string pressure_topic_ = "/auv/pressure/scaled2";
    std::string mag_topic_ = "/auv/imu/magnetic_field";
    double dvl_timeout_ = 0.25;
    double pressure_timeout_ = 0.25;
    double mag_timeout_ = 0.5;
    double dvl_velocity_cov_ = 4e-4;
    double dvl_innovation_gate_sigma_ = 5.0;
    double dvl_b_init_cov_ = 1e-8;
    double pressure_cov_ = 1e4;
    double pressure_b_init_cov_ = 1e4;
    double pressure_innovation_gate_sigma_ = 0.0;
    V3D camera_init_T_in_world_ = V3D::Zero();
    M3D camera_init_R_in_world_ = M3D::Identity();
    double pressure_fluid_density_ = 1025.0;
    double pressure_gravity_ = 9.80665;
    double pressure_surface_pressure_ = 101325.0;
    double pressure_surface_z_ = 0.0;
    double pressure_reference_sensor_z_world_ = 0.0;
    static constexpr int kPressureReferenceSamples = 20;
    double pressure_init_sum_ = 0.0;
    int pressure_init_samples_collected_ = 0;
    bool pressure_ref_finalized_ = false;
    bool pressure_reference_pose_ready_ = false;
    double mag_cov_ = 1849.0;
    double mag_heading_cov_floor_ = 1e-6;
    double mag_innovation_gate_sigma_ = 3.0;
    static constexpr int kMagReferenceSamples = 20;
    V3D mag_reference_sum_local_ = V3D::Zero();
    std::vector<V3D> mag_reference_samples_local_;
    int mag_reference_sample_count_ = 0;
    bool mag_reference_ready_ = false;
    // Fixed when reference collection starts; magnetometer updates rotate only about this axis.
    V3D mag_vertical_local_ = V3D::UnitZ();
    V3D mag_reference_local_ = V3D::Zero();
    V3D mag_horizontal_reference_local_ = V3D::Zero();
    double mag_reference_heading_variance_ = 0.0;
    V3D dvl_T_ = V3D::Zero();
    M3D dvl_R_ = M3D::Identity();
    V3D pressure_T_ = V3D::Zero();
    M3D mag_R_BM_ = M3D::Identity();
    V3D mag_hard_iron_ = V3D::Zero();
    M3D mag_soft_iron_ = M3D::Identity();
    std::string mag_debug_csv_path_;
    std::ofstream mag_debug_csv_;

    mutable std::mutex mutex_;
    std::deque<DvlMsg::ConstSharedPtr> dvl_buffer_;
    std::deque<PressureMsg::ConstSharedPtr> pressure_buffer_;
    std::deque<MagMsg::ConstSharedPtr> mag_buffer_;
    LateMeasurementCounts late_measurement_counts_;
    double last_timestamp_dvl_ = -1.0;
    double last_timestamp_pressure_ = -1.0;
    double last_timestamp_mag_ = -1.0;
    rclcpp::Subscription<DvlMsg>::SharedPtr sub_dvl_;
    rclcpp::Subscription<PressureMsg>::SharedPtr sub_pressure_;
    rclcpp::Subscription<MagMsg>::SharedPtr sub_mag_;
};

#endif
