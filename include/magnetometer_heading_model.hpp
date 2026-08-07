#ifndef MAGNETOMETER_HEADING_MODEL_HPP
#define MAGNETOMETER_HEADING_MODEL_HPP

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "IKFoM_toolkit/mtk/src/mtkmath.hpp"

namespace underwater_fastlio::magnetometer
{

inline double wrap_pi(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

struct HeadingObservation
{
    bool valid = false;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    double a = 0.0;
    double b = 0.0;
    double denominator = 0.0;
    double innovation = 0.0;
    Eigen::RowVector3d iteration_jacobian = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d observation_jacobian = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d magnetic_jacobian = Eigen::RowVector3d::Zero();
};

// R maps body vectors into the local estimation frame. IKFoM uses the right
// perturbation R_true = R_est Exp(delta_theta^), so H = -d(nu)/d(delta_est).
inline HeadingObservation evaluate(const Eigen::Matrix3d &R,
                                   const Eigen::Vector3d &magnetic_body,
                                   const Eigen::Vector3d &horizontal_reference_local,
                                   const Eigen::Vector3d &vertical_local = Eigen::Vector3d::UnitZ())
{
    HeadingObservation out;
    if (!R.allFinite() || !magnetic_body.allFinite() ||
        !horizontal_reference_local.allFinite() || !vertical_local.allFinite())
    {
        return out;
    }

    const double n_norm = vertical_local.norm();
    const double h_norm = horizontal_reference_local.norm();
    if (n_norm <= 1e-12 || h_norm <= 1e-12)
    {
        return out;
    }

    const Eigen::Vector3d n = vertical_local / n_norm;
    const Eigen::Vector3d h0 = horizontal_reference_local / h_norm;
    out.g = R.transpose() * n;
    out.p = R.transpose() * h0;
    const Eigen::Vector3d magnetic_cross_p = magnetic_body.cross(out.p);
    out.a = out.g.dot(magnetic_cross_p);
    out.b = magnetic_body.dot(out.p);
    out.denominator = out.a * out.a + out.b * out.b;
    if (!std::isfinite(out.denominator) || out.denominator <= 1e-18)
    {
        return out;
    }

    out.innovation = wrap_pi(std::atan2(out.a, out.b));
    out.iteration_jacobian =
        (out.b * ((out.g.dot(magnetic_body)) * out.p - out.b * out.g).transpose() -
         out.a * magnetic_cross_p.transpose()) /
        out.denominator;
    out.observation_jacobian = -out.iteration_jacobian;
    out.magnetic_jacobian =
        (out.b * out.p.cross(out.g).transpose() - out.a * out.p.transpose()) /
        out.denominator;
    out.valid = out.iteration_jacobian.allFinite() &&
                out.observation_jacobian.allFinite() &&
                out.magnetic_jacobian.allFinite() && std::isfinite(out.innovation);
    return out;
}

inline bool reference_sample_is_inlier(const Eigen::Vector3d &sample_local,
                                       const Eigen::Vector3d &running_mean_local,
                                       int accepted_count)
{
    constexpr double kNormGateRatio = 0.35;
    constexpr double kAngleGateRad = 20.0 * M_PI / 180.0;
    if (!sample_local.allFinite() || sample_local.norm() <= 1e-12)
    {
        return false;
    }
    if (accepted_count < 5 || running_mean_local.norm() <= 1e-12)
    {
        return true;
    }
    if (std::abs(sample_local.norm() - running_mean_local.norm()) /
            running_mean_local.norm() > kNormGateRatio)
    {
        return false;
    }
    const double cosine = std::clamp(sample_local.normalized().dot(running_mean_local.normalized()),
                                     -1.0, 1.0);
    return std::acos(cosine) <= kAngleGateRad;
}

inline Eigen::VectorXd constrained_gain(const Eigen::MatrixXd &P,
                                        const Eigen::RowVectorXd &H,
                                        double innovation_variance,
                                        const Eigen::Vector3d &heading_direction_body,
                                        int attitude_index = 3,
                                        int gyro_bias_index = 15)
{
    Eigen::VectorXd constrained = Eigen::VectorXd::Zero(P.rows());
    if (P.rows() != P.cols() || H.size() != P.rows() ||
        innovation_variance <= 0.0 || !std::isfinite(innovation_variance))
    {
        return constrained;
    }
    const Eigen::VectorXd unconstrained = P * H.transpose() / innovation_variance;
    const Eigen::Vector3d g = heading_direction_body.normalized();
    constrained.segment<3>(attitude_index) =
        g * g.dot(unconstrained.segment<3>(attitude_index));
    constrained.segment<3>(gyro_bias_index) =
        g * g.dot(unconstrained.segment<3>(gyro_bias_index));
    return constrained;
}

inline Eigen::MatrixXd joseph_covariance(const Eigen::MatrixXd &P,
                                         const Eigen::RowVectorXd &H,
                                         const Eigen::VectorXd &constrained_gain_vector,
                                         double measurement_variance)
{
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
    const Eigen::MatrixXd A = I - constrained_gain_vector * H;
    Eigen::MatrixXd out = A * P * A.transpose() +
                          measurement_variance * constrained_gain_vector *
                              constrained_gain_vector.transpose();
    return 0.5 * (out + out.transpose()).eval();
}

inline Eigen::Matrix3d right_reset_jacobian(const Eigen::Vector3d &delta_theta)
{
    // This is the same SO(3) covariance reset used by IKFoM after boxplus.
    MTK::vect<3, double> segment;
    segment << delta_theta.x(), delta_theta.y(), delta_theta.z();
    return MTK::A_matrix(segment).transpose();
}

inline Eigen::MatrixXd transport_attitude_covariance(const Eigen::MatrixXd &P,
                                                      const Eigen::Vector3d &delta_theta,
                                                      int attitude_index = 3)
{
    Eigen::MatrixXd transport = Eigen::MatrixXd::Identity(P.rows(), P.cols());
    transport.block<3, 3>(attitude_index, attitude_index) =
        right_reset_jacobian(delta_theta);
    Eigen::MatrixXd out = transport * P * transport.transpose();
    return 0.5 * (out + out.transpose()).eval();
}

inline bool covariance_is_psd(const Eigen::MatrixXd &P, double tolerance = 1e-10)
{
    if (P.rows() != P.cols() || !P.allFinite())
    {
        return false;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(0.5 * (P + P.transpose()));
    return solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() >= -tolerance;
}

}  // namespace underwater_fastlio::magnetometer

#endif
