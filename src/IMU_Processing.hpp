#include <algorithm>
#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <csignal>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <functional>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ekf = esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom>;
  using TimedUpdateCallback = std::function<bool(std::size_t, Ekf &)>;

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  // void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset(double start_timestamp, const sensor_msgs::msg::Imu::ConstSharedPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_initial_cov(const V3D &b_g, const V3D &b_a, double grav);
  // Auxiliary-bias initial covariances (added with DVL/pressure fusion). Locking
  // these in noiseless-IMU sim mode is what stops LiDAR scan-match residuals from
  // bleeding into b_dvl/b_pressure via the P-inverse cross-correlations and
  // breaking DVL/pressure observability mid-bag.
  void set_initial_aux_cov(const V3D &b_dvl, double b_pressure);
  void set_gravity(const double gravity_m_s2);
  bool IsInitialized() const;
  Eigen::Matrix<double, process_noise_ikfom::DOF, process_noise_ikfom::DOF> Q;
  void Process(const MeasureGroup &meas,
               Ekf &kf_state,
               PointCloudXYZI::Ptr pcl_un_,
               const std::vector<double> &update_times = {},
               const TimedUpdateCallback &timed_update = {});

  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  V3D init_cov_bias_gyr;
  V3D init_cov_bias_acc;
  double init_cov_grav;
  V3D init_cov_b_dvl;
  double init_cov_b_pressure;
  double first_lidar_time;

 private:
  void IMU_init(const MeasureGroup &meas,
                Ekf &kf_state,
                int &N);
  void UndistortPcl(const MeasureGroup &meas,
                    Ekf &kf_state,
                    PointCloudXYZI &pcl_in_out,
                    const std::vector<double> &update_times,
                    const TimedUpdateCallback &timed_update);
  void UndistortPclFastLio(const MeasureGroup &meas,
                           Ekf &kf_state,
                           PointCloudXYZI &pcl_in_out);
  bool ReconstructContinuousDeskewPoses(
      const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> &imu_msgs,
      double scan_begin_time,
      double scan_end_time,
      const state_ikfom &scan_end_state);

  PointCloudXYZI::Ptr cur_pcl_un_;
  // sensor_msgs::ImuConstPtr last_imu_;
  sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double gravity_m_s2_;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), gravity_m_s2_(G_m_s2), start_timestamp_(-1), last_lidar_end_time_(-1.0)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  init_cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
  init_cov_bias_acc = V3D(0.001, 0.001, 0.001);
  init_cov_grav = 0.00001;
  // Defaults preserve the previous hard-coded values in init_P (1e-8 for b_dvl,
  // 1e4 for b_pressure). These are loose enough that on the all-sensor run the
  // LiDAR update was free to absorb scan-vs-pressure z disagreement into a
  // ~21 cm equivalent pressure bias and to drift b_dvl by O(1e-3) m/s.
  init_cov_b_dvl = V3D(1e-8, 1e-8, 1e-8);
  init_cov_b_pressure = 1e4;
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::msg::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  last_lidar_end_time_ = -1.0;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::msg::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::set_initial_cov(const V3D &b_g, const V3D &b_a, double grav)
{
  init_cov_bias_gyr = b_g;
  init_cov_bias_acc = b_a;
  init_cov_grav = grav;
}

void ImuProcess::set_initial_aux_cov(const V3D &b_dvl, double b_pressure)
{
  init_cov_b_dvl = b_dvl;
  init_cov_b_pressure = b_pressure;
}

void ImuProcess::set_gravity(const double gravity_m_s2)
{
  gravity_m_s2_ = gravity_m_s2;
}

bool ImuProcess::IsInitialized() const
{
  return !imu_need_init_;
}

void ImuProcess::IMU_init(
    const MeasureGroup &meas,
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom> &kf_state,
    int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * gravity_m_s2_);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_cov_bias_gyr[0];
  init_P(16,16) = init_cov_bias_gyr[1];
  init_P(17,17) = init_cov_bias_gyr[2];
  init_P(18,18) = init_cov_bias_acc[0];
  init_P(19,19) = init_cov_bias_acc[1];
  init_P(20,20) = init_cov_bias_acc[2];
  init_P(21,21) = init_P(22,22) = init_cov_grav;
  init_P(23,23) = init_cov_b_dvl[0];
  init_P(24,24) = init_cov_b_dvl[1];
  init_P(25,25) = init_cov_b_dvl[2];
  init_P(26,26) = init_cov_b_pressure;
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

bool ImuProcess::ReconstructContinuousDeskewPoses(
    const std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> &imu_msgs,
    double scan_begin_time,
    double scan_end_time,
    const state_ikfom &scan_end_state)
{
  struct ImuSegment
  {
    double begin = 0.0;
    double end = 0.0;
    V3D gyro = V3D::Zero();
    V3D acc = V3D::Zero();
  };

  if (imu_msgs.size() < 2 || scan_end_time <= scan_begin_time)
  {
    return false;
  }

  std::vector<ImuSegment> segments;
  double covered_time = scan_begin_time;
  V3D previous_gyro = V3D::Zero();
  V3D previous_acc = V3D::Zero();
  bool have_previous_input = false;
  constexpr double time_epsilon = 1e-9;

  auto append_segment = [&](double begin, double end,
                            const V3D &gyro, const V3D &acc)
  {
    if (end <= begin + time_epsilon)
    {
      return;
    }
    segments.push_back({begin, end, gyro, acc});
    covered_time = end;
  };

  for (auto it = imu_msgs.begin(); it < imu_msgs.end() - 1; ++it)
  {
    const auto &head = *it;
    const auto &tail = *(it + 1);
    const double head_time = rclcpp::Time(head->header.stamp).seconds();
    const double tail_time = rclcpp::Time(tail->header.stamp).seconds();
    if (tail_time <= head_time + time_epsilon)
    {
      continue;
    }

    V3D gyro;
    gyro << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    V3D acc;
    acc << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
           0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
           0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
    acc *= gravity_m_s2_ / mean_acc.norm();

    previous_gyro = gyro;
    previous_acc = acc;
    have_previous_input = true;

    if (tail_time <= scan_begin_time + time_epsilon)
    {
      continue;
    }
    if (head_time >= scan_end_time - time_epsilon)
    {
      break;
    }

    const double segment_begin = std::max(scan_begin_time, head_time);
    const double segment_end = std::min(scan_end_time, tail_time);
    if (segment_begin > covered_time + time_epsilon)
    {
      append_segment(covered_time, segment_begin, gyro, acc);
    }
    append_segment(std::max(covered_time, segment_begin), segment_end, gyro, acc);
    if (covered_time >= scan_end_time - time_epsilon)
    {
      break;
    }
  }

  if (have_previous_input && covered_time < scan_end_time - time_epsilon)
  {
    append_segment(covered_time, scan_end_time, previous_gyro, previous_acc);
  }
  if (segments.empty() ||
      segments.front().begin > scan_begin_time + time_epsilon ||
      segments.back().end < scan_end_time - time_epsilon)
  {
    return false;
  }

  struct KinematicState
  {
    M3D rotation = M3D::Identity();
    V3D velocity = V3D::Zero();
    V3D position = V3D::Zero();
  };

  std::vector<KinematicState> states(segments.size() + 1);
  std::vector<V3D> segment_acc_world(segments.size(), V3D::Zero());
  std::vector<V3D> segment_gyro_body(segments.size(), V3D::Zero());
  states.back().rotation = scan_end_state.rot.toRotationMatrix();
  states.back().velocity = scan_end_state.vel;
  states.back().position = scan_end_state.pos;
  const V3D gravity(scan_end_state.grav[0],
                    scan_end_state.grav[1],
                    scan_end_state.grav[2]);
  const V3D gyro_bias(scan_end_state.bg[0],
                      scan_end_state.bg[1],
                      scan_end_state.bg[2]);
  const V3D acc_bias(scan_end_state.ba[0],
                     scan_end_state.ba[1],
                     scan_end_state.ba[2]);

  for (std::size_t reverse_index = segments.size(); reverse_index > 0; --reverse_index)
  {
    const std::size_t i = reverse_index - 1;
    const double dt = segments[i].end - segments[i].begin;
    const V3D omega = segments[i].gyro - gyro_bias;
    const V3D specific_force = segments[i].acc - acc_bias;
    states[i].rotation = states[i + 1].rotation * Exp(omega, -dt);
    const V3D acc_begin = states[i].rotation * specific_force + gravity;
    const V3D acc_end = states[i + 1].rotation * specific_force + gravity;
    const V3D acc_world = 0.5 * (acc_begin + acc_end);
    states[i].velocity = states[i + 1].velocity - acc_world * dt;
    states[i].position = states[i + 1].position -
                         states[i + 1].velocity * dt +
                         0.5 * acc_world * dt * dt;
    segment_acc_world[i] = acc_world;
    segment_gyro_body[i] = omega;
  }

  IMUpose.clear();
  IMUpose.reserve(states.size());
  IMUpose.push_back(set_pose6d(
      0.0, segment_acc_world.front(), segment_gyro_body.front(),
      states.front().velocity, states.front().position, states.front().rotation));
  for (std::size_t i = 0; i < segments.size(); ++i)
  {
    IMUpose.push_back(set_pose6d(
        segments[i].end - scan_begin_time,
        segment_acc_world[i], segment_gyro_body[i],
        states[i + 1].velocity, states[i + 1].position,
        states[i + 1].rotation));
  }
  return true;
}

void ImuProcess::UndistortPcl(
    const MeasureGroup &meas,
    Ekf &kf_state,
    PointCloudXYZI &pcl_out,
    const std::vector<double> &update_times,
    const TimedUpdateCallback &timed_update)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = rclcpp::Time(v_imu.front()->header.stamp).seconds();
  const double &imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each IMU point and auxiliary timestamp ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  input_ikfom in;
  bool have_input = false;
  bool auxiliary_state_updated = false;
  std::size_t update_index = 0;

  auto record_pose = [&](double timestamp)
  {
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);
    for (int i = 0; i < 3; ++i) acc_s_last[i] += imu_state.grav[i];
    const double offset_time = timestamp - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offset_time, acc_s_last, angvel_last,
                                imu_state.vel, imu_state.pos,
                                imu_state.rot.toRotationMatrix()));
  };

  // Advance one constant-IMU-input segment, splitting it only where an
  // asynchronous measurement belongs. With no events this performs the same
  // single predict call as the original FAST-LIO2 propagation.
  auto propagate_segment = [&](double segment_start, double segment_end,
                               bool record_segment_end)
  {
    double current_time = segment_start;
    constexpr double time_epsilon = 1e-9;
    while (update_index < update_times.size() &&
           update_times[update_index] <= segment_end + time_epsilon)
    {
      const double event_time =
          std::clamp(update_times[update_index], current_time, segment_end);
      double dt = event_time - current_time;
      if (dt > 0.0) kf_state.predict(dt, Q, in);
      current_time = event_time;

      const double grouped_timestamp = update_times[update_index];
      do
      {
        if (timed_update)
        {
          auxiliary_state_updated =
              timed_update(update_index, kf_state) || auxiliary_state_updated;
        }
        ++update_index;
      }
      while (update_index < update_times.size() &&
             update_times[update_index] <= grouped_timestamp + time_epsilon &&
             update_times[update_index] <= segment_end + time_epsilon);

      if (current_time >= pcl_beg_time - time_epsilon &&
          current_time < segment_end - time_epsilon)
      {
        record_pose(current_time);
      }
    }

    double remaining_dt = segment_end - current_time;
    if (std::abs(remaining_dt) > time_epsilon)
    {
      kf_state.predict(remaining_dt, Q, in);
    }
    if (record_segment_end) record_pose(segment_end);
  };

  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp < last_lidar_end_time_)    continue;

    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);


    acc_avr     = acc_avr * gravity_m_s2_ / mean_acc.norm(); // - state_inout.ba;

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    have_input = true;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    const double segment_start =
        head_stamp < last_lidar_end_time_ ? last_lidar_end_time_ : head_stamp;
    propagate_segment(segment_start, tail_stamp, true);
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  if (have_input)
  {
    if (pcl_end_time >= imu_end_time)
    {
      propagate_segment(imu_end_time, pcl_end_time, false);
    }
    else
    {
      // Preserve FAST-LIO2's original positive frame-end extrapolation.
      double final_dt = imu_end_time - pcl_end_time;
      kf_state.predict(final_dt, Q, in);
    }
  }

  imu_state = kf_state.get_x();
  if (auxiliary_state_updated)
  {
    ReconstructContinuousDeskewPoses(
        v_imu, pcl_beg_time, pcl_end_time, imu_state);
  }
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      const double dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::UndistortPclFastLio(
    const MeasureGroup &meas,
    Ekf &kf_state,
    PointCloudXYZI &pcl_out)
{
  // Keep the no-auxiliary path identical to FAST-LIO2's scan-bounded IMU
  // propagation and backward point-cloud undistortion.
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double pcl_beg_time = meas.lidar_beg_time;
  const double pcl_end_time = meas.lidar_end_time;

  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last,
                              imu_state.vel, imu_state.pos,
                              imu_state.rot.toRotationMatrix()));

  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  double dt = 0.0;
  input_ikfom in;
  bool have_input = false;

  for (auto it_imu = v_imu.begin(); it_imu < v_imu.end() - 1; ++it_imu)
  {
    const auto &head = *it_imu;
    const auto &tail = *(it_imu + 1);
    const double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    const double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp < last_lidar_end_time_) continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                  0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                  0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
               0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
               0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
    acc_avr *= gravity_m_s2_ / mean_acc.norm();

    dt = head_stamp < last_lidar_end_time_
             ? tail_stamp - last_lidar_end_time_
             : tail_stamp - head_stamp;
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    have_input = true;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);
    for (int i = 0; i < 3; ++i) acc_s_last[i] += imu_state.grav[i];
    const double offset_time = tail_stamp - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offset_time, acc_s_last, angvel_last,
                                imu_state.vel, imu_state.pos,
                                imu_state.rot.toRotationMatrix()));
  }

  if (have_input)
  {
    const double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);
  }

  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  if (pcl_out.points.empty()) return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); --it_kp)
  {
    const auto head = it_kp - 1;
    const auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    acc_imu << VEC_FROM_ARRAY(tail->acc);
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);

    for (; it_pcl->curvature / 1000.0 > head->offset_time; --it_pcl)
    {
      dt = it_pcl->curvature / 1000.0 - head->offset_time;
      const M3D R_i(R_imu * Exp(angvel_avr, dt));
      const V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      const V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt -
                     imu_state.pos);
      const V3D P_compensate =
          imu_state.offset_R_L_I.conjugate() *
          (imu_state.rot.conjugate() *
               (R_i * (imu_state.offset_R_L_I * P_i +
                       imu_state.offset_T_L_I) +
                T_ei) -
           imu_state.offset_T_L_I);
      it_pcl->x = P_compensate.x();
      it_pcl->y = P_compensate.y();
      it_pcl->z = P_compensate.z();
      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::Process(
    const MeasureGroup &meas,
    Ekf &kf_state,
    PointCloudXYZI::Ptr cur_pcl_un_,
    const std::vector<double> &update_times,
    const TimedUpdateCallback &timed_update)
{
  if(meas.imu.empty()) {return;};
  assert(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;

    last_imu_   = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(gravity_m_s2_ / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
    }

    return;
  }

  if (update_times.empty())
  {
    UndistortPclFastLio(meas, kf_state, *cur_pcl_un_);
  }
  else
  {
    UndistortPcl(meas, kf_state, *cur_pcl_un_, update_times, timed_update);
  }
}
