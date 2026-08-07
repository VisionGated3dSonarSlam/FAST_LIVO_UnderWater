#ifndef DVL_MEASUREMENT_MODEL_HPP
#define DVL_MEASUREMENT_MODEL_HPP

#include <Eigen/Core>

#include "use-ikfom.hpp"

namespace underwater_fastlio::dvl
{

constexpr int kVelocityIndex = 12;
constexpr int kAttitudeIndex = 3;
constexpr int kGyroBiasIndex = 15;
constexpr int kDvlBiasIndex = 23;

inline Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d out;
    out << 0.0, -v.z(), v.y(),
           v.z(), 0.0, -v.x(),
          -v.y(), v.x(), 0.0;
    return out;
}

struct Model
{
    Eigen::Vector3d imu_origin_velocity_vehicle = Eigen::Vector3d::Zero();
    Eigen::Vector3d dvl_origin_velocity_vehicle = Eigen::Vector3d::Zero();
    Eigen::Vector3d prediction_dvl = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, state_ikfom::DOF> H =
        Eigen::Matrix<double, 3, state_ikfom::DOF>::Zero();
};

// R_VD rotates DVL-frame vectors into the vehicle/IMU frame. Its transpose
// therefore maps the physical velocity prediction into the native DVL frame.
inline Model evaluate(const state_ikfom &state,
                      const Eigen::Vector3d &raw_gyro_vehicle,
                      const Eigen::Matrix3d &R_VD,
                      const Eigen::Vector3d &p_D_vehicle)
{
    Model model;
    const Eigen::Matrix3d R_LV = state.rot.toRotationMatrix();
    const Eigen::Matrix3d C_DV = R_VD.transpose();
    const Eigen::Vector3d velocity_local(state.vel[0], state.vel[1], state.vel[2]);
    const Eigen::Vector3d gyro_bias_vehicle(state.bg[0], state.bg[1], state.bg[2]);
    const Eigen::Vector3d dvl_bias_dvl(state.b_dvl[0], state.b_dvl[1], state.b_dvl[2]);

    model.imu_origin_velocity_vehicle = R_LV.transpose() * velocity_local;
    const Eigen::Vector3d angular_velocity_vehicle =
        raw_gyro_vehicle - gyro_bias_vehicle;
    model.dvl_origin_velocity_vehicle =
        model.imu_origin_velocity_vehicle + angular_velocity_vehicle.cross(p_D_vehicle);
    model.prediction_dvl = C_DV * model.dvl_origin_velocity_vehicle + dvl_bias_dvl;

    // IKFoM applies a right attitude perturbation, R_true = R Exp(dtheta^).
    // H is the derivative of h(x), paired with the innovation z - h(x).
    model.H.template block<3, 3>(0, kAttitudeIndex) =
        C_DV * skew(model.imu_origin_velocity_vehicle);
    model.H.template block<3, 3>(0, kVelocityIndex) = C_DV * R_LV.transpose();
    model.H.template block<3, 3>(0, kGyroBiasIndex) = C_DV * skew(p_D_vehicle);
    model.H.template block<3, 3>(0, kDvlBiasIndex).setIdentity();
    return model;
}

}  // namespace underwater_fastlio::dvl

#endif
