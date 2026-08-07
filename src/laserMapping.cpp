// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: FAST-LIO contributors

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <cmath>
#include <thread>
#include <csignal>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include "auxiliary_sensor_fusion.hpp"
#include "observability_manager.hpp"
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV_DEFAULT (0.001)
#define PUBFRAME_PERIOD     (20)
double LASER_POINT_COV_XY = LASER_POINT_COV_DEFAULT;
double LASER_POINT_COV_Z = LASER_POINT_COV_DEFAULT;

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   pcd_save_en = false, path_en = true;
/**************************/

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, world_frame;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
V3D imu_gyro_scale(1.0, 1.0, 1.0);  // per-axis multiplicative correction for raw gyro (e.g. 1/0.74 for a known scale error)
double init_b_gyr_cov = 0.0001, init_b_acc_cov = 0.001, init_grav_cov = 0.00001;
double init_b_dvl_cov = 1e-8, init_b_pressure_cov = 1e4;
bool noiseless_imu = false;
double imu_orientation_cov = 3.0461742e-6;  // (0.1 deg)^2
double imu_orientation_gate_sigma = 0.0;
bool imu_orientation_ref_ready = false;
Eigen::Quaterniond imu_orientation_ref = Eigen::Quaterniond::Identity();
bool accel_attitude_ref_ready = false;
double accel_attitude_cov = 1.2184697e-3;  // (2 deg)^2
double accel_attitude_norm_gate = 2.0;
ObservabilityManager obs_manager;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
double last_processed_time = -1.0, lidar_timeout = 0.25, imu_rate_hz = 100.0;
double gravity_m_s2 = G_m_s2;
int    effct_feat_num = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
using MainEkf = esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom>;
MainEkf kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());
AuxiliarySensorFusion *g_joint_aux_fusion = nullptr;
AuxiliarySensorFusion::JointMeasurementSet g_joint_aux_measurements;
bool g_joint_aux_update_active = false;

void SigHandle(int sig)
{
    (void)sig;
    flg_exit = true;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    obs_manager.notify_sonar_scan(cur_time);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    publish_count ++;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->angular_velocity.x *= imu_gyro_scale.x();
    msg->angular_velocity.y *= imu_gyro_scale.y();
    msg->angular_velocity.z *= imu_gyro_scale.z();

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
double expected_imu_period()
{
    return 1.0 / (imu_rate_hz > 1.0 ? imu_rate_hz : 1.0);
}

double expected_imu_timeout()
{
    return 2.5 * expected_imu_period();
}

double imu_only_packet_duration()
{
    return std::max(5.0 * expected_imu_period(), std::min(lidar_timeout, 0.25));
}

bool sync_packages(MeasureGroup &meas)
{
    while (!lidar_pushed && !lidar_buffer.empty() && last_processed_time > 0.0 &&
           time_buffer.front() < last_processed_time - 1e-4)
    {
        lidar_buffer.pop_front();
        time_buffer.pop_front();
    }

    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

bool sync_imu_only_packages(MeasureGroup &meas)
{
    if (!lidar_buffer.empty() || lidar_pushed || imu_buffer.empty()) {
        return false;
    }

    const double latest_imu_time = get_time_sec(imu_buffer.back()->header.stamp);
    if (last_timestamp_lidar > 0.0 && latest_imu_time - last_timestamp_lidar < lidar_timeout) {
        return false;
    }
    if (last_processed_time > 0.0 && latest_imu_time <= last_processed_time + 1e-6) {
        return false;
    }
    const double first_imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    meas.lidar.reset(new PointCloudXYZI());
    double packet_begin_time = last_processed_time > 0.0 ? last_processed_time : first_imu_time;
    if (last_processed_time > 0.0 && first_imu_time > packet_begin_time + expected_imu_timeout())
    {
        packet_begin_time = first_imu_time;
    }
    const double target_packet_end_time = packet_begin_time + imu_only_packet_duration();
    if (latest_imu_time < target_packet_end_time - 1e-6)
    {
        return false;
    }

    meas.lidar_beg_time = packet_begin_time;
    double last_included_imu_time = packet_begin_time;

    while (!imu_buffer.empty())
    {
        const double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if (last_processed_time > 0.0 && imu_time <= last_processed_time + 1e-6)
        {
            imu_buffer.pop_front();
            continue;
        }
        if (imu_time > target_packet_end_time + 1e-6) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
        last_included_imu_time = imu_time;
    }

    if (meas.imu.empty())
    {
        return false;
    }
    meas.lidar_end_time = last_included_imu_time;
    lidar_end_time = meas.lidar_end_time;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en || pcd_save_en)
    {
        if (pcd_save_en)
        {
            const int save_size = feats_down_body->points.size();
            PointCloudXYZI::Ptr compact_cloud_world(new PointCloudXYZI(save_size, 1));
            for (int i = 0; i < save_size; i++)
            {
                RGBpointBodyToWorld(&feats_down_body->points[i],
                                    &compact_cloud_world->points[i]);
            }
            *pcl_wait_pub += *compact_cloud_world;
        }

        if(scan_pub_en)
        {
            PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
            int size = laserCloudFullRes->points.size();
            PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

            for (int i = 0; i < size; i++)
            {
                RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                    &laserCloudWorld->points[i]);
            }

            sensor_msgs::msg::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
            // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
            laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
            laserCloudmsg.header.frame_id = "camera_init";
            pubLaserCloudFull->publish(laserCloudmsg);
            publish_count -= PUBFRAME_PERIOD;
        }
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = "camera_init";
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void update_state_outputs()
{
    state_point = kf.get_x();
    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];
}

const char *g_publish_mode = "init";

bool apply_imu_orientation_update(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg)
{
    if (!imu_msg || imu_msg->orientation_covariance[0] < 0.0)
    {
        return false;
    }

    const auto &q_msg = imu_msg->orientation;
    Eigen::Quaterniond q_meas(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    if (!std::isfinite(q_meas.w()) || !std::isfinite(q_meas.x()) ||
        !std::isfinite(q_meas.y()) || !std::isfinite(q_meas.z()) ||
        q_meas.norm() < 1e-6)
    {
        return false;
    }
    q_meas.normalize();
    if (!imu_orientation_ref_ready)
    {
        imu_orientation_ref = q_meas;
        imu_orientation_ref_ready = true;
        return false;
    }

    Eigen::Quaterniond q_relative = imu_orientation_ref.conjugate() * q_meas;
    q_relative.normalize();
    state_ikfom state = kf.get_x();
    const M3D R_err =
        state.rot.toRotationMatrix().transpose() * q_relative.toRotationMatrix();
    const V3D residual = Log(R_err);

    MainEkf::cov P = kf.get_P();
    Eigen::Matrix<double, 3, state_ikfom::DOF> H =
        Eigen::Matrix<double, 3, state_ikfom::DOF>::Zero();
    H.block<3, 3>(0, 3).setIdentity();
    const M3D R = M3D::Identity() * std::max(1e-12, imu_orientation_cov);
    const M3D S = H * P * H.transpose() + R;
    Eigen::LDLT<M3D> ldlt(S);
    if (ldlt.info() != Eigen::Success)
    {
        return false;
    }
    if (imu_orientation_gate_sigma > 0.0)
    {
        const double nis = residual.dot(ldlt.solve(residual));
        if (!std::isfinite(nis) ||
            nis > 3.0 * imu_orientation_gate_sigma * imu_orientation_gate_sigma)
        {
            return false;
        }
    }

    const Eigen::Matrix<double, state_ikfom::DOF, 3> K =
        P * H.transpose() * ldlt.solve(M3D::Identity());
    const Eigen::Matrix<double, state_ikfom::DOF, 1> dx = K * residual;
    if (!dx.allFinite())
    {
        return false;
    }
    state.boxplus(dx);

    const MainEkf::cov I = MainEkf::cov::Identity();
    const MainEkf::cov KH = K * H;
    MainEkf::cov P_new =
        ((I - KH) * P * (I - KH).transpose() + K * R * K.transpose()).eval();
    P_new = ((P_new + P_new.transpose()) * 0.5).eval();
    kf.change_x(state);
    kf.change_P(P_new);
    return true;
}

bool apply_accel_attitude_update(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg)
{
    if (!imu_msg)
    {
        return false;
    }

    state_ikfom state = kf.get_x();
    V3D acc_meas(imu_msg->linear_acceleration.x,
                 imu_msg->linear_acceleration.y,
                 imu_msg->linear_acceleration.z);
    acc_meas -= V3D(state.ba[0], state.ba[1], state.ba[2]);
    const double acc_norm = acc_meas.norm();
    if (!std::isfinite(acc_norm) || acc_norm < 1e-6 ||
        std::abs(acc_norm - gravity_m_s2) > accel_attitude_norm_gate)
    {
        return false;
    }
    const bool initialize_from_filter_gravity = !accel_attitude_ref_ready;
    accel_attitude_ref_ready = true;
    V3D grav_local(state.grav[0], state.grav[1], state.grav[2]);
    if (grav_local.norm() < 1e-6)
    {
        grav_local = V3D(0.0, 0.0, -gravity_m_s2);
    }
    const V3D predicted =
        state.rot.toRotationMatrix().transpose() * (-grav_local.normalized());
    // IMU initialization has already estimated gravity from the stationary
    // startup window. Condition the attitude covariance on that same prior
    // once, without allowing the first post-init (possibly moving) sample to
    // rotate the state. Later calls use the current measured acceleration.
    const V3D measured = initialize_from_filter_gravity
        ? predicted
        : acc_meas / acc_norm;
    if (!measured.allFinite() || !predicted.allFinite())
    {
        return false;
    }

    Eigen::Quaterniond predicted_to_measured =
        Eigen::Quaterniond::FromTwoVectors(predicted, measured);
    predicted_to_measured.normalize();
    V3D residual = -Log(predicted_to_measured.toRotationMatrix());
    residual -= predicted * residual.dot(predicted);
    if (!residual.allFinite())
    {
        return false;
    }

    MainEkf::cov P = kf.get_P();
    Eigen::Matrix<double, 3, state_ikfom::DOF> H =
        Eigen::Matrix<double, 3, state_ikfom::DOF>::Zero();
    H.block<3, 3>(0, 3).setIdentity();
    const M3D R = M3D::Identity() * std::max(1e-8, accel_attitude_cov);
    const M3D S = H * P * H.transpose() + R;
    Eigen::LDLT<M3D> ldlt(S);
    if (ldlt.info() != Eigen::Success)
    {
        return false;
    }
    const Eigen::Matrix<double, state_ikfom::DOF, 3> K =
        P * H.transpose() * ldlt.solve(M3D::Identity());
    const Eigen::Matrix<double, state_ikfom::DOF, 1> dx = K * residual;
    if (!dx.allFinite())
    {
        return false;
    }
    state.boxplus(dx);

    const MainEkf::cov I = MainEkf::cov::Identity();
    const MainEkf::cov KH = K * H;
    MainEkf::cov P_new =
        ((I - KH) * P * (I - KH).transpose() + K * R * K.transpose()).eval();
    P_new = ((P_new + P_new.transpose()) * 0.5).eval();
    kf.change_x(state);
    kf.change_P(P_new);
    return true;
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }
    pubOdomAftMapped->publish(odomAftMapped);

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.child_frame_id = "body";
    trans.header.stamp = get_ros_time(lidar_end_time);
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            effct_feat_num ++;
        }
    }
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    const int max_joint_aux_rows =
        (g_joint_aux_update_active && g_joint_aux_fusion) ? g_joint_aux_measurements.max_rows() : 0;
    const int max_rows = effct_feat_num + max_joint_aux_rows;
    if (max_rows < 1)
    {
        ekfom_data.valid = false;
        return;
    }
    // Preserve the original FAST-LIO2 12-column LiDAR Jacobian and solver
    // exactly when this scan has no DVL or pressure rows. The expanded state
    // Jacobian is used only for a real joint auxiliary correction.
    const int jacobian_cols = max_joint_aux_rows > 0 ? state_ikfom::DOF : 12;
    ekfom_data.h_x = MatrixXd::Zero(max_rows, jacobian_cols);
    ekfom_data.h = Eigen::VectorXd::Zero(max_rows);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;

        // The point-to-plane residual constrains motion along the map normal:
        // horizontal/seabed normals mainly constrain z, vertical-wall normals
        // mainly constrain xy. Whiten each row so LiDAR z can be tuned
        // independently from LiDAR xy without changing IKFoM's scalar-R API.
        const double nz2 = std::clamp(static_cast<double>(norm_p.z) * static_cast<double>(norm_p.z), 0.0, 1.0);
        const double point_cov = std::max(1e-12, LASER_POINT_COV_XY * (1.0 - nz2) + LASER_POINT_COV_Z * nz2);
        const double inv_sigma = 1.0 / std::sqrt(point_cov);
        ekfom_data.h_x.block<1, 12>(i, 0) *= inv_sigma;
        ekfom_data.h(i) *= inv_sigma;
    }

    int total_rows = effct_feat_num;
    if (g_joint_aux_update_active && g_joint_aux_fusion)
    {
        total_rows = g_joint_aux_fusion->append_joint_measurement_rows(
            g_joint_aux_measurements, s, kf.get_P(),
            ekfom_data.h_x, ekfom_data.h, total_rows);
    }
    if (total_rows < 1)
    {
        ekfom_data.valid = false;
        return;
    }
    ekfom_data.h_x.conservativeResize(total_rows, Eigen::NoChange);
    ekfom_data.h.conservativeResize(total_rows);
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/points_raw");
        this->declare_parameter<string>("common.imu_topic", "/imu/data");
        this->declare_parameter<string>("common.world_frame", "world");
        this->declare_parameter<vector<double>>("common.world_to_camera_init_T", {0.0, 0.0, 0.0});
        this->declare_parameter<vector<double>>("common.world_to_camera_init_R",
                             {1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0});
        this->declare_parameter<double>("common.imu_rate_hz", 100.0);
        this->declare_parameter<double>("common.lidar_timeout", 0.25);
        this->declare_parameter<double>("common.gravity_m_s2", G_m_s2);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("mapping.init_b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.init_b_acc_cov", 0.001);
        this->declare_parameter<double>("mapping.init_grav_cov", 0.00001);
        this->declare_parameter<double>("mapping.imu_orientation_cov", 3.0461742e-6);
        this->declare_parameter<double>("mapping.imu_orientation_gate_sigma", 0.0);
        this->declare_parameter<double>("mapping.accel_attitude_cov", 1.2184697e-3);
        this->declare_parameter<double>("mapping.accel_attitude_norm_gate", 2.0);
        this->declare_parameter<bool>("mapping.noiseless_imu", false);
        this->declare_parameter<vector<double>>("mapping.imu_gyro_scale", {1.0, 1.0, 1.0});
        ObservabilityManager::declare_parameters(*this);
        this->declare_parameter<double>("mapping.laser_point_cov", LASER_POINT_COV_DEFAULT);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        aux_fusion_.declare_parameters(*this);

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/points_raw");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/imu/data");
        this->get_parameter_or<string>("common.world_frame", world_frame, "world");
        std::vector<double> world_to_camera_init_T;
        std::vector<double> world_to_camera_init_R;
        this->get_parameter_or<vector<double>>("common.world_to_camera_init_T",
                            world_to_camera_init_T,
                            {0.0, 0.0, 0.0});
        this->get_parameter_or<vector<double>>("common.world_to_camera_init_R",
                            world_to_camera_init_R,
                            {1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0});
        this->get_parameter_or<double>("common.imu_rate_hz", imu_rate_hz, 100.0);
        this->get_parameter_or<double>("common.lidar_timeout", lidar_timeout, 0.25);
        this->get_parameter_or<double>("common.gravity_m_s2", gravity_m_s2, G_m_s2);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("mapping.init_b_gyr_cov",init_b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.init_b_acc_cov",init_b_acc_cov,0.001);
        this->get_parameter_or<double>("mapping.init_grav_cov",init_grav_cov,0.00001);
        this->get_parameter_or<double>("mapping.imu_orientation_cov", imu_orientation_cov, 3.0461742e-6);
        this->get_parameter_or<double>("mapping.imu_orientation_gate_sigma", imu_orientation_gate_sigma, 0.0);
        this->get_parameter_or<double>("mapping.accel_attitude_cov", accel_attitude_cov, 1.2184697e-3);
        this->get_parameter_or<double>("mapping.accel_attitude_norm_gate", accel_attitude_norm_gate, 2.0);
        this->get_parameter_or<bool>("mapping.noiseless_imu",noiseless_imu,false);
        {
            vector<double> scale_vec = {1.0, 1.0, 1.0};
            this->get_parameter_or<vector<double>>("mapping.imu_gyro_scale", scale_vec, {1.0, 1.0, 1.0});
            if (scale_vec.size() == 3)
                imu_gyro_scale = V3D(scale_vec[0], scale_vec[1], scale_vec[2]);
            else
                RCLCPP_WARN(this->get_logger(), "mapping.imu_gyro_scale must have 3 values. Using [1,1,1].");
        }
        double legacy_laser_point_cov = LASER_POINT_COV_DEFAULT;
        this->get_parameter_or<double>("mapping.laser_point_cov", legacy_laser_point_cov, double(LASER_POINT_COV_DEFAULT));
        this->declare_parameter<double>("mapping.laser_point_cov_xy", legacy_laser_point_cov);
        this->declare_parameter<double>("mapping.laser_point_cov_z", legacy_laser_point_cov);
        this->get_parameter_or<double>("mapping.laser_point_cov_xy", LASER_POINT_COV_XY, legacy_laser_point_cov);
        this->get_parameter_or<double>("mapping.laser_point_cov_z", LASER_POINT_COV_Z, legacy_laser_point_cov);
        LASER_POINT_COV_XY = std::max(1e-12, LASER_POINT_COV_XY);
        LASER_POINT_COV_Z = std::max(1e-12, LASER_POINT_COV_Z);
        imu_orientation_cov = std::max(1e-12, imu_orientation_cov);
        imu_orientation_gate_sigma = std::max(0.0, imu_orientation_gate_sigma);
        accel_attitude_cov = std::max(1e-8, accel_attitude_cov);
        accel_attitude_norm_gate = std::max(0.0, accel_attitude_norm_gate);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        aux_fusion_.load_parameters(*this);
        g_joint_aux_fusion = &aux_fusion_;
        if (imu_rate_hz <= 0.0)
        {
            RCLCPP_WARN(this->get_logger(), "common.imu_rate_hz must be positive. Falling back to 100 Hz.");
            imu_rate_hz = 100.0;
        }
        if (gravity_m_s2 <= 0.0)
        {
            RCLCPP_WARN(this->get_logger(), "common.gravity_m_s2 must be positive. Falling back to %.2f m/s^2.", G_m_s2);
            gravity_m_s2 = G_m_s2;
        }
        if (noiseless_imu)
        {
            // Simulation-only mode: IMU has zero bias/noise. Freeze ba/bg/grav so
            // LiDAR residuals cannot rewrite them. Only valid for simulated IMUs.
            b_acc_cov = std::min(b_acc_cov, 1e-10);
            b_gyr_cov = std::min(b_gyr_cov, 1e-10);
            init_b_acc_cov = 1e-10;
            init_b_gyr_cov = 1e-10;
            init_grav_cov = 1e-10;
        }
        // else: init_b_acc_cov / init_b_gyr_cov / init_grav_cov already loaded from YAML above.
        const double disabled_aux_cov = 1e-12;
        init_b_dvl_cov = aux_fusion_.dvl_enabled() ? aux_fusion_.dvl_b_init_cov() : disabled_aux_cov;
        init_b_pressure_cov = aux_fusion_.pressure_enabled() ? 1e4 : disabled_aux_cov;

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="camera_init";

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));

        if (extrinT.size() != 3)
        {
            RCLCPP_WARN(this->get_logger(),
                        "mapping.extrinsic_T must have 3 values. Using zero translation.");
            extrinT = {0.0, 0.0, 0.0};
        }
        if (extrinR.size() != 9)
        {
            RCLCPP_WARN(this->get_logger(),
                        "mapping.extrinsic_R must have 9 values. Using identity rotation.");
            extrinR = {1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0};
        }
        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gravity(gravity_m_s2);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
        p_imu->set_initial_cov(V3D(init_b_gyr_cov, init_b_gyr_cov, init_b_gyr_cov),
                               V3D(init_b_acc_cov, init_b_acc_cov, init_b_acc_cov),
                               init_grav_cov);
        p_imu->set_initial_aux_cov(V3D(init_b_dvl_cov, init_b_dvl_cov, init_b_dvl_cov),
                                   init_b_pressure_cov);
        obs_manager.load_parameters(*this, b_gyr_cov, b_acc_cov);

        fill(epsi, epsi + state_ikfom::DOF, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** ROS subscribe initialization ***/
        sensor_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        processing_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions sensor_options;
        sensor_options.callback_group = sensor_callback_group_;

        if (lid_topic.empty())
        {
            RCLCPP_WARN(this->get_logger(),
                        "common.lid_topic is empty. LiDAR subscription is disabled; node will run IMU-only odometry.");
        }
        else
        {
            auto lidar_qos = rclcpp::QoS(rclcpp::KeepLast(200000));
            lidar_qos.best_effort();
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                lid_topic, lidar_qos, standard_pcl_cbk, sensor_options);
        }
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, rclcpp::QoS(rclcpp::KeepLast(200000)), imu_cbk, sensor_options);
        aux_fusion_.create_subscriptions(*this, sensor_callback_group_);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

        if (world_to_camera_init_T.size() != 3)
        {
            RCLCPP_WARN(this->get_logger(),
                        "common.world_to_camera_init_T must have 3 values. Using zero translation.");
            world_to_camera_init_T = {0.0, 0.0, 0.0};
        }
        Eigen::Matrix3d world_to_camera_init_rot = Eigen::Matrix3d::Identity();
        if (world_to_camera_init_R.size() == 9)
        {
            world_to_camera_init_rot << world_to_camera_init_R[0], world_to_camera_init_R[1], world_to_camera_init_R[2],
                                        world_to_camera_init_R[3], world_to_camera_init_R[4], world_to_camera_init_R[5],
                                        world_to_camera_init_R[6], world_to_camera_init_R[7], world_to_camera_init_R[8];
        }
        else
        {
            RCLCPP_WARN(this->get_logger(),
                        "common.world_to_camera_init_R must have 9 values. Using identity rotation.");
        }
        Eigen::Quaterniond world_to_camera_init_quat(world_to_camera_init_rot);
        world_to_camera_init_quat.normalize();

        geometry_msgs::msg::TransformStamped world_to_camera_init;
        world_to_camera_init.header.stamp = this->get_clock()->now();
        world_to_camera_init.header.frame_id = world_frame;
        world_to_camera_init.child_frame_id = "camera_init";
        world_to_camera_init.transform.translation.x = world_to_camera_init_T[0];
        world_to_camera_init.transform.translation.y = world_to_camera_init_T[1];
        world_to_camera_init.transform.translation.z = world_to_camera_init_T[2];
        world_to_camera_init.transform.rotation.x = world_to_camera_init_quat.x();
        world_to_camera_init.transform.rotation.y = world_to_camera_init_quat.y();
        world_to_camera_init.transform.rotation.z = world_to_camera_init_quat.z();
        world_to_camera_init.transform.rotation.w = world_to_camera_init_quat.w();
        static_tf_broadcaster_->sendTransform(world_to_camera_init);

        // Inform the pressure model how camera_init sits in World, so pressure
        // constrains true World-vertical depth rather than tilted local z.
        aux_fusion_.set_camera_init_pose_in_world(
            V3D(world_to_camera_init_T[0], world_to_camera_init_T[1], world_to_camera_init_T[2]),
            world_to_camera_init_rot);

        //------------------------------------------------------------------------------------------------------
        // Drive processing from wall time so rosbag replay can drain buffered
        // messages even after simulated /clock stops publishing. Sensor callbacks
        // are isolated in their own callback group and only fill mutex-protected
        // buffers; the EKF state remains owned by this processing callback.
        timer_ = this->create_wall_timer(std::chrono::milliseconds(1),
                                         std::bind(&LaserMappingNode::timer_callback, this),
                                         processing_callback_group_);

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

private:
    void timer_callback()
    {
        bool imu_only_measure = false;
        bool has_measurement = false;
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            has_measurement = sync_packages(Measures);
            if (!has_measurement)
            {
                imu_only_measure = sync_imu_only_packages(Measures);
                has_measurement = imu_only_measure;
            }
        }

        if(has_measurement)
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                if (!imu_only_measure)
                {
                    return;
                }
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            const double process_begin_time = last_processed_time > 0.0 ? last_processed_time : Measures.lidar_beg_time;
            if (!noiseless_imu)
            {
                const double now = Measures.lidar_end_time;
                const double dyn_bg = obs_manager.bg_cov(now);
                const double dyn_ba = obs_manager.ba_cov(now);
                p_imu->set_gyr_bias_cov(V3D(dyn_bg, dyn_bg, dyn_bg));
                p_imu->set_acc_bias_cov(V3D(dyn_ba, dyn_ba, dyn_ba));
            }
            if (!p_imu->IsInitialized())
            {
                p_imu->Process(Measures, kf, feats_undistort);
                last_processed_time = Measures.lidar_end_time;
                update_state_outputs();
                return;
            }

            AuxiliarySensorFusion::UpdateSummary aux_summary;
            auto apply_aux_updates = [&]() {
                aux_summary.merge(aux_fusion_.process_interval(
                    process_begin_time, Measures.lidar_end_time, Measures.imu, kf));
                aux_summary.merge(aux_fusion_.process_magnetometer_interval(
                    process_begin_time, Measures.lidar_end_time, kf));
                aux_fusion_.warn_timeouts(*this, Measures.lidar_end_time);
            };

            if (imu_only_measure)
            {
                auto propagate_to = [&](double target_time) {
                    return p_imu->PartialPropagate(target_time, kf, Measures.imu);
                };
                // With no scan correction, aux measurements are part of the
                // propagation prior itself. Fuse them at their own timestamps;
                // batching early magnetometer samples at the packet end can
                // inject a false startup heading when the vehicle is moving.
                aux_summary = aux_fusion_.process_interval_interleaved(
                    process_begin_time, Measures.lidar_end_time, Measures.imu, kf,
                    propagate_to);
                p_imu->PartialPropagate(Measures.lidar_end_time, kf, Measures.imu);
                if (!Measures.imu.empty())
                {
                    // Interpret a valid absolute IMU quaternion relative to
                    // startup so its external frame cannot inject an initial
                    // heading or tilt into FAST-LIO's local estimation frame.
                    apply_imu_orientation_update(Measures.imu.back());
                    apply_accel_attitude_update(Measures.imu.back());
                }
                last_processed_time = Measures.lidar_end_time;
                aux_fusion_.warn_timeouts(*this, Measures.lidar_end_time);
                update_state_outputs();
                if (lid_topic.empty())
                {
                    // Intentional no-lidar mode. Startup already reported this once.
                }
                else if (last_timestamp_lidar <= 0.0)
                {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                         "No LiDAR messages received on '%s'. Running IMU-only odometry.",
                                         lid_topic.c_str());
                }
                else
                {
                    const double lidar_gap = Measures.lidar_end_time - last_timestamp_lidar;
                    if (lidar_gap >= lidar_timeout)
                    {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                             "No LiDAR scan for %.2f s (timeout %.2f s). Running IMU-only odometry.",
                                             lidar_gap, lidar_timeout);
                    }
                }
                g_publish_mode = aux_summary.updated() ? "aux_only" : "imu_only";
                publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
                return;
            }

            p_imu->Process(Measures, kf, feats_undistort);
            last_processed_time = Measures.lidar_end_time;
            if (!Measures.imu.empty())
            {
                apply_imu_orientation_update(Measures.imu.back());
                apply_accel_attitude_update(Measures.imu.back());
            }
            update_state_outputs();

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                apply_aux_updates();
                update_state_outputs();
                RCLCPP_WARN(this->get_logger(), "No point, publish IMU-only odometry for this scan.\n");
                g_publish_mode = aux_summary.updated() ? "aux_only" : "no_points";
                publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                apply_aux_updates();
                update_state_outputs();
                g_publish_mode = "kdtree_init";
                publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                apply_aux_updates();
                update_state_outputs();
                RCLCPP_WARN(this->get_logger(), "Too few points, publish IMU-only odometry for this scan.\n");
                g_publish_mode = aux_summary.updated() ? "aux_only" : "few_points";
                publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            state_ikfom state_before_lidar = kf.get_x();
            g_joint_aux_measurements = aux_fusion_.take_joint_measurements(
                process_begin_time, Measures.lidar_end_time, Measures.imu);
            aux_summary = aux_fusion_.summarize_joint_measurements(
                g_joint_aux_measurements, state_before_lidar, kf.get_P());
            g_joint_aux_update_active = g_joint_aux_measurements.max_rows() > 0;
            kf.update_iterated_dyn_share_modified(1.0, solve_H_time);
            g_joint_aux_update_active = false;
            // Magnetometer is deliberately excluded from the joint scan/DVL/
            // pressure residual. Apply one constrained scalar heading update to
            // the corrected scan state, before publishing and map insertion.
            aux_summary.merge(aux_fusion_.process_magnetometer_interval(
                process_begin_time, Measures.lidar_end_time, kf));
            update_state_outputs();

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            g_publish_mode = "lidar_update";
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull_);
        }
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    AuxiliarySensorFusion aux_fusion_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::CallbackGroup::SharedPtr sensor_callback_group_;
    rclcpp::CallbackGroup::SharedPtr processing_callback_group_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0;
    double deltaT, deltaR;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[state_ikfom::DOF] = {0.001};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    auto node = std::make_shared<LaserMappingNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    return 0;
}
