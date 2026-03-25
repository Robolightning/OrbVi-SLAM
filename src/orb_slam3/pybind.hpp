#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <nanobind/ndarray.h>

namespace obvi_orbslam3 {

namespace nb = nanobind;

// Forward declarations of upstream ORB-SLAM3 types.
namespace ORB_SLAM3 {
class System;
class Settings;
class ConfigParser;
}  // namespace ORB_SLAM3

using Sensor = ORB_SLAM3::System::eSensor;
using CameraType = ORB_SLAM3::Settings::CameraType;

struct PoseSE3f {
  std::array<double, 16> matrix{};
  std::array<double, 3> translation{};
  std::array<double, 4> quaternion_wxyz{};

  PoseSE3f();
};

struct TrackingResult {
  PoseSE3f pose;
  int tracking_state = -1;
  double timestamp = 0.0;
};

class PySettings {
public:
  PySettings(const std::string &config_file, Sensor sensor);

  CameraType camera_type() const;
  std::pair<int, int> new_im_size() const;
  float fps() const;
  bool rgb() const;
  bool need_to_undistort() const;
  bool need_to_resize() const;
  bool need_to_rectify() const;

  float bf() const;
  float b() const;
  float th_depth() const;

  float noise_gyro() const;
  float noise_acc() const;
  float gyro_walk() const;
  float acc_walk() const;
  float imu_frequency() const;
  bool insert_kfs_when_lost() const;

  float depth_map_factor() const;

  int n_features() const;
  int n_levels() const;
  float init_th_fast() const;
  float min_th_fast() const;
  float scale_factor() const;

  float key_frame_size() const;
  float key_frame_line_width() const;
  float graph_line_width() const;
  float point_size() const;
  float camera_size() const;
  float camera_line_width() const;
  float view_point_x() const;
  float view_point_y() const;
  float view_point_z() const;
  float view_point_f() const;
  float image_viewer_scale() const;

  std::string atlas_load_file() const;
  std::string atlas_save_file() const;

  float th_far_points() const;

private:
  std::unique_ptr<ORB_SLAM3::Settings> settings_;
};

class PyConfigParser {
public:
  PyConfigParser() = default;

  bool parse_config_file(const std::string &config_file);

private:
  ORB_SLAM3::ConfigParser *parser_ = nullptr;
  std::unique_ptr<ORB_SLAM3::ConfigParser> owned_parser_;
};

class PySystem {
public:
  PySystem(const std::string &voc_file,
           const std::string &settings_file,
           Sensor sensor,
           bool use_viewer = false,
           int init_fr = 0,
           const std::string &sequence = std::string());

  ~PySystem();

  PySystem(const PySystem &) = delete;
  PySystem &operator=(const PySystem &) = delete;
  PySystem(PySystem &&) noexcept = default;
  PySystem &operator=(PySystem &&) noexcept = default;

  TrackingResult track_monocular(
      const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &image,
      double timestamp,
      const std::string &filename = std::string());

  TrackingResult track_stereo(
      const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &left,
      const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &right,
      double timestamp,
      const std::string &filename = std::string());

  TrackingResult track_rgbd(
      const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &rgb,
      const nb::ndarray<nb::numpy, const float, nb::c_contig> &depth,
      double timestamp,
      const std::string &filename = std::string());

  void activate_localization_mode();
  void deactivate_localization_mode();
  bool map_changed();
  void reset();
  void reset_active_map();
  void shutdown();
  bool is_shutdown() const;

  void save_trajectory_tum(const std::string &filename);
  void save_keyframe_trajectory_tum(const std::string &filename);
  void save_trajectory_euroc(const std::string &filename);
  void save_keyframe_trajectory_euroc(const std::string &filename);
  void save_trajectory_kitti(const std::string &filename);
  void save_debug_data(int ini_idx);

  int get_tracking_state();
  double get_time_from_imu_init();
  bool is_lost();
  bool is_finished();
  void change_dataset();
  float get_image_scale();

private:
  std::unique_ptr<ORB_SLAM3::System> system_;
};

}  // namespace obvi_orbslam3