#include "orb_slam3_pybind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <opencv2/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Config.h"
#include "Settings.h"
#include "System.h"

namespace nb = nanobind;
using namespace obvi_orbslam3;

namespace {

template <typename T>
cv::Mat ndarray_to_mat_8u(const nb::ndarray<nb::numpy, const T, nb::c_contig> &arr,
                          const char *name) {
  static_assert(std::is_same_v<T, std::uint8_t> || std::is_same_v<T, float>,
                "Unsupported ndarray element type");

  if constexpr (std::is_same_v<T, std::uint8_t>) {
    if (arr.ndim() != 2 && arr.ndim() != 3) {
      throw nb::value_error(std::string(name) +
                            " must be a 2D grayscale or 3D color array");
    }

    const int rows = static_cast<int>(arr.shape(0));
    const int cols = static_cast<int>(arr.shape(1));

    int channels = 1;
    int type = CV_8UC1;

    if (arr.ndim() == 3) {
      channels = static_cast<int>(arr.shape(2));
      if (channels == 1) {
        type = CV_8UC1;
      } else if (channels == 3) {
        type = CV_8UC3;
      } else {
        throw nb::value_error(std::string(name) +
                              " must have 1 or 3 channels in the last dimension");
      }
    }

    return cv::Mat(rows,
                   cols,
                   type,
                   const_cast<std::uint8_t *>(arr.data()),
                   static_cast<size_t>(arr.stride(0)));
  } else {
    if (arr.ndim() != 2) {
      throw nb::value_error(std::string(name) + " must be a 2D depth array");
    }

    const int rows = static_cast<int>(arr.shape(0));
    const int cols = static_cast<int>(arr.shape(1));
    return cv::Mat(rows,
                   cols,
                   CV_32FC1,
                   const_cast<float *>(arr.data()),
                   static_cast<size_t>(arr.stride(0)));
  }
}

PoseSE3f pose_from_se3f(const Sophus::SE3f &Tcw) {
  PoseSE3f out;

  Eigen::Matrix4f m = Tcw.matrix();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      out.matrix[r * 4 + c] = static_cast<double>(m(r, c));
    }
  }

  const Eigen::Vector3f t = Tcw.translation();
  out.translation = {static_cast<double>(t.x()),
                     static_cast<double>(t.y()),
                     static_cast<double>(t.z())};

  const Eigen::Quaternionf q(Tcw.unit_quaternion());
  out.quaternion_wxyz = {static_cast<double>(q.w()),
                         static_cast<double>(q.x()),
                         static_cast<double>(q.y()),
                         static_cast<double>(q.z())};

  return out;
}

TrackingResult make_tracking_result(const Sophus::SE3f &pose, int state, double ts) {
  TrackingResult out;
  out.pose = pose_from_se3f(pose);
  out.tracking_state = state;
  out.timestamp = ts;
  return out;
}

}  // namespace

PoseSE3f::PoseSE3f() {
  matrix = {1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0};
  translation = {0.0, 0.0, 0.0};
  quaternion_wxyz = {1.0, 0.0, 0.0, 0.0};
}

PySettings::PySettings(const std::string &config_file, Sensor sensor)
    : settings_(std::make_unique<ORB_SLAM3::Settings>(config_file,
                                                      static_cast<int>(sensor))) {}

CameraType PySettings::camera_type() const {
  return settings_->cameraType();
}

std::pair<int, int> PySettings::new_im_size() const {
  const cv::Size s = settings_->newImSize();
  return {s.width, s.height};
}

float PySettings::fps() const { return settings_->fps(); }
bool PySettings::rgb() const { return settings_->rgb(); }
bool PySettings::need_to_undistort() const { return settings_->needToUndistort(); }
bool PySettings::need_to_resize() const { return settings_->needToResize(); }
bool PySettings::need_to_rectify() const { return settings_->needToRectify(); }

float PySettings::bf() const { return settings_->bf(); }
float PySettings::b() const { return settings_->b(); }
float PySettings::th_depth() const { return settings_->thDepth(); }

float PySettings::noise_gyro() const { return settings_->noiseGyro(); }
float PySettings::noise_acc() const { return settings_->noiseAcc(); }
float PySettings::gyro_walk() const { return settings_->gyroWalk(); }
float PySettings::acc_walk() const { return settings_->accWalk(); }
float PySettings::imu_frequency() const { return settings_->imuFrequency(); }
bool PySettings::insert_kfs_when_lost() const { return settings_->insertKFsWhenLost(); }

float PySettings::depth_map_factor() const { return settings_->depthMapFactor(); }

int PySettings::n_features() const { return settings_->nFeatures(); }
int PySettings::n_levels() const { return settings_->nLevels(); }
float PySettings::init_th_fast() const { return settings_->initThFAST(); }
float PySettings::min_th_fast() const { return settings_->minThFAST(); }
float PySettings::scale_factor() const { return settings_->scaleFactor(); }

float PySettings::key_frame_size() const { return settings_->keyFrameSize(); }
float PySettings::key_frame_line_width() const { return settings_->keyFrameLineWidth(); }
float PySettings::graph_line_width() const { return settings_->graphLineWidth(); }
float PySettings::point_size() const { return settings_->pointSize(); }
float PySettings::camera_size() const { return settings_->cameraSize(); }
float PySettings::camera_line_width() const { return settings_->cameraLineWidth(); }
float PySettings::view_point_x() const { return settings_->viewPointX(); }
float PySettings::view_point_y() const { return settings_->viewPointY(); }
float PySettings::view_point_z() const { return settings_->viewPointZ(); }
float PySettings::view_point_f() const { return settings_->viewPointF(); }
float PySettings::image_viewer_scale() const { return settings_->imageViewerScale(); }

std::string PySettings::atlas_load_file() const { return settings_->atlasLoadFile(); }
std::string PySettings::atlas_save_file() const { return settings_->atlasSaveFile(); }

float PySettings::th_far_points() const { return settings_->thFarPoints(); }

bool PyConfigParser::parse_config_file(const std::string &config_file) {
  owned_parser_ = std::make_unique<ORB_SLAM3::ConfigParser>();
  parser_ = owned_parser_.get();

  std::string path = config_file;
  return parser_->ParseConfigFile(path);
}

PySystem::PySystem(const std::string &voc_file,
                   const std::string &settings_file,
                   Sensor sensor,
                   bool use_viewer,
                   int init_fr,
                   const std::string &sequence)
    : system_(std::make_unique<ORB_SLAM3::System>(voc_file,
                                                   settings_file,
                                                   sensor,
                                                   use_viewer,
                                                   init_fr,
                                                   sequence)) {}

PySystem::~PySystem() {
  try {
    if (system_ && !system_->isShutDown()) {
      system_->Shutdown();
    }
  } catch (...) {
    // Never throw from destructor.
  }
}

TrackingResult PySystem::track_monocular(
    const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &image,
    double timestamp,
    const std::string &filename) {
  nb::gil_scoped_release release;
  cv::Mat im = ndarray_to_mat_8u(image, "image");
  Sophus::SE3f pose = system_->TrackMonocular(im, timestamp, {}, filename);
  const int state = system_->GetTrackingState();
  return make_tracking_result(pose, state, timestamp);
}

TrackingResult PySystem::track_stereo(
    const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &left,
    const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &right,
    double timestamp,
    const std::string &filename) {
  nb::gil_scoped_release release;
  cv::Mat im_left = ndarray_to_mat_8u(left, "left");
  cv::Mat im_right = ndarray_to_mat_8u(right, "right");
  Sophus::SE3f pose = system_->TrackStereo(im_left, im_right, timestamp, {}, filename);
  const int state = system_->GetTrackingState();
  return make_tracking_result(pose, state, timestamp);
}

TrackingResult PySystem::track_rgbd(
    const nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig> &rgb,
    const nb::ndarray<nb::numpy, const float, nb::c_contig> &depth,
    double timestamp,
    const std::string &filename) {
  nb::gil_scoped_release release;
  cv::Mat im = ndarray_to_mat_8u(rgb, "rgb");
  cv::Mat depthmap = ndarray_to_mat_8u(depth, "depth");
  Sophus::SE3f pose = system_->TrackRGBD(im, depthmap, timestamp, {}, filename);
  const int state = system_->GetTrackingState();
  return make_tracking_result(pose, state, timestamp);
}

void PySystem::activate_localization_mode() {
  nb::gil_scoped_release release;
  system_->ActivateLocalizationMode();
}

void PySystem::deactivate_localization_mode() {
  nb::gil_scoped_release release;
  system_->DeactivateLocalizationMode();
}

bool PySystem::map_changed() {
  nb::gil_scoped_release release;
  return system_->MapChanged();
}

void PySystem::reset() {
  nb::gil_scoped_release release;
  system_->Reset();
}

void PySystem::reset_active_map() {
  nb::gil_scoped_release release;
  system_->ResetActiveMap();
}

void PySystem::shutdown() {
  nb::gil_scoped_release release;
  if (system_ && !system_->isShutDown()) {
    system_->Shutdown();
  }
}

bool PySystem::is_shutdown() const {
  return system_ ? system_->isShutDown() : true;
}

void PySystem::save_trajectory_tum(const std::string &filename) {
  nb::gil_scoped_release release;
  system_->SaveTrajectoryTUM(filename);
}

void PySystem::save_keyframe_trajectory_tum(const std::string &filename) {
  nb::gil_scoped_release release;
  system_->SaveKeyFrameTrajectoryTUM(filename);
}

void PySystem::save_trajectory_euroc(const std::string &filename) {
  nb::gil_scoped_release release;
  system_->SaveTrajectoryEuRoC(filename);
}

void PySystem::save_keyframe_trajectory_euroc(const std::string &filename) {
  nb::gil_scoped_release release;
  system_->SaveKeyFrameTrajectoryEuRoC(filename);
}

void PySystem::save_trajectory_kitti(const std::string &filename) {
  nb::gil_scoped_release release;
  system_->SaveTrajectoryKITTI(filename);
}

void PySystem::save_debug_data(int ini_idx) {
  nb::gil_scoped_release release;
  system_->SaveDebugData(ini_idx);
}

int PySystem::get_tracking_state() {
  return system_->GetTrackingState();
}

double PySystem::get_time_from_imu_init() {
  return system_->GetTimeFromIMUInit();
}

bool PySystem::is_lost() {
  return system_->isLost();
}

bool PySystem::is_finished() {
  return system_->isFinished();
}

void PySystem::change_dataset() {
  nb::gil_scoped_release release;
  system_->ChangeDataset();
}

float PySystem::get_image_scale() {
  return system_->GetImageScale();
}

NB_MODULE(obvi_orb_slam3, m) {
  m.doc() = "Nanobind bindings for ORB-SLAM3 tailored for ObVi-SLAM integration.";

  nb::enum_<Sensor>(m, "Sensor", "Input sensor mode used by ORB-SLAM3.")
      .value("MONOCULAR", Sensor::MONOCULAR, "Monocular camera input.")
      .value("STEREO", Sensor::STEREO, "Stereo camera input.")
      .value("RGBD", Sensor::RGBD, "RGB-D camera input.")
      .value("IMU_MONOCULAR", Sensor::IMU_MONOCULAR, "Monocular + IMU input.")
      .value("IMU_STEREO", Sensor::IMU_STEREO, "Stereo + IMU input.")
      .value("IMU_RGBD", Sensor::IMU_RGBD, "RGB-D + IMU input.")
      .export_values();

  nb::enum_<CameraType>(m, "CameraType", "Camera model type parsed from settings.")
      .value("PinHole", CameraType::PinHole, "Pinhole camera model.")
      .value("Rectified", CameraType::Rectified, "Rectified stereo model.")
      .value("KannalaBrandt", CameraType::KannalaBrandt, "Kannala-Brandt model.")
      .export_values();

  nb::class_<PoseSE3f>(m, "PoseSE3f", "Rigid-body pose represented as SE(3) in float precision.")
      .def(nb::init<>(), "Create an identity pose.")
      .def_rw("matrix", &PoseSE3f::matrix, "Homogeneous 4x4 transform matrix in row-major order.")
      .def_rw("translation", &PoseSE3f::translation, "Translation vector [x, y, z].")
      .def_rw("quaternion_wxyz", &PoseSE3f::quaternion_wxyz,
              "Rotation quaternion [w, x, y, z].");

  nb::class_<TrackingResult>(m, "TrackingResult",
                             "Result returned from a single tracking call.")
      .def(nb::init<>(), "Create an empty tracking result.")
      .def_rw("pose", &TrackingResult::pose, "Estimated camera pose.")
      .def_rw("tracking_state", &TrackingResult::tracking_state,
              "Raw tracking state reported by ORB-SLAM3 after the call.")
      .def_rw("timestamp", &TrackingResult::timestamp,
              "Timestamp passed to the tracking call.");

  nb::class_<PySettings>(m, "Settings",
                         "Thin wrapper around ORB-SLAM3 Settings loaded from a YAML file.")
      .def(nb::init<const std::string &, Sensor>(),
           nb::arg("config_file"),
           nb::arg("sensor"),
           "Load ORB-SLAM3 settings from a configuration file.")
      .def("camera_type", &PySettings::camera_type, "Return the parsed camera model type.")
      .def("new_im_size", &PySettings::new_im_size,
           "Return the resized image size as (width, height).")
      .def("fps", &PySettings::fps, "Return the configured camera FPS.")
      .def("rgb", &PySettings::rgb, "Return whether images are interpreted as RGB.")
      .def("need_to_undistort", &PySettings::need_to_undistort,
           "Return whether undistortion is required.")
      .def("need_to_resize", &PySettings::need_to_resize,
           "Return whether images are resized before tracking.")
      .def("need_to_rectify", &PySettings::need_to_rectify,
           "Return whether stereo rectification is required.")
      .def("bf", &PySettings::bf, "Return baseline times focal length.")
      .def("b", &PySettings::b, "Return stereo baseline.")
      .def("th_depth", &PySettings::th_depth,
           "Return the depth threshold used by ORB-SLAM3.")
      .def("noise_gyro", &PySettings::noise_gyro, "Return gyro noise model parameter.")
      .def("noise_acc", &PySettings::noise_acc, "Return accelerometer noise model parameter.")
      .def("gyro_walk", &PySettings::gyro_walk, "Return gyro random walk parameter.")
      .def("acc_walk", &PySettings::acc_walk, "Return accelerometer random walk parameter.")
      .def("imu_frequency", &PySettings::imu_frequency, "Return IMU frequency.")
      .def("insert_kfs_when_lost", &PySettings::insert_kfs_when_lost,
           "Return whether keyframes are inserted while tracking is lost.")
      .def("depth_map_factor", &PySettings::depth_map_factor,
           "Return the depth map conversion factor.")
      .def("n_features", &PySettings::n_features, "Return the ORB feature count.")
      .def("n_levels", &PySettings::n_levels, "Return the ORB pyramid level count.")
      .def("init_th_fast", &PySettings::init_th_fast, "Return the FAST threshold used at initialization.")
      .def("min_th_fast", &PySettings::min_th_fast, "Return the minimum FAST threshold.")
      .def("scale_factor", &PySettings::scale_factor, "Return the ORB pyramid scale factor.")
      .def("key_frame_size", &PySettings::key_frame_size, "Return keyframe rendering size.")
      .def("key_frame_line_width", &PySettings::key_frame_line_width, "Return keyframe rendering line width.")
      .def("graph_line_width", &PySettings::graph_line_width, "Return graph rendering line width.")
      .def("point_size", &PySettings::point_size, "Return map point rendering size.")
      .def("camera_size", &PySettings::camera_size, "Return camera rendering size.")
      .def("camera_line_width", &PySettings::camera_line_width, "Return camera rendering line width.")
      .def("view_point_x", &PySettings::view_point_x, "Return viewer viewpoint X coordinate.")
      .def("view_point_y", &PySettings::view_point_y, "Return viewer viewpoint Y coordinate.")
      .def("view_point_z", &PySettings::view_point_z, "Return viewer viewpoint Z coordinate.")
      .def("view_point_f", &PySettings::view_point_f, "Return viewer viewpoint focal distance.")
      .def("image_viewer_scale", &PySettings::image_viewer_scale, "Return image viewer scale.")
      .def("atlas_load_file", &PySettings::atlas_load_file, "Return the atlas load file path.")
      .def("atlas_save_file", &PySettings::atlas_save_file, "Return the atlas save file path.")
      .def("th_far_points", &PySettings::th_far_points, "Return the far point threshold.");

  nb::class_<PyConfigParser>(m, "ConfigParser",
                             "Thin wrapper around ORB-SLAM3 configuration parser.")
      .def(nb::init<>(), "Create a parser instance.")
      .def("parse_config_file", &PyConfigParser::parse_config_file,
           nb::arg("config_file"),
           "Parse an ORB-SLAM3 configuration file and return True on success.");

  nb::class_<PySystem>(m, "System",
                       "High-level ORB-SLAM3 system wrapper used by the project.")
      .def(nb::init<const std::string &,
                    const std::string &,
                    Sensor,
                    bool,
                    int,
                    const std::string &>(),
           nb::arg("voc_file"),
           nb::arg("settings_file"),
           nb::arg("sensor"),
           nb::arg("use_viewer") = false,
           nb::arg("init_fr") = 0,
           nb::arg("sequence") = std::string(),
           "Create and initialize the ORB-SLAM3 system.")
      .def("track_monocular", &PySystem::track_monocular,
           nb::arg("image"),
           nb::arg("timestamp"),
           nb::arg("filename") = std::string(),
           "Track a monocular image and return pose plus raw tracking state.")
      .def("track_stereo", &PySystem::track_stereo,
           nb::arg("left"),
           nb::arg("right"),
           nb::arg("timestamp"),
           nb::arg("filename") = std::string(),
           "Track a stereo pair and return pose plus raw tracking state.")
      .def("track_rgbd", &PySystem::track_rgbd,
           nb::arg("rgb"),
           nb::arg("depth"),
           nb::arg("timestamp"),
           nb::arg("filename") = std::string(),
           "Track an RGB-D frame and return pose plus raw tracking state.")
      .def("activate_localization_mode", &PySystem::activate_localization_mode,
           "Stop local mapping and switch to localization-only mode.")
      .def("deactivate_localization_mode", &PySystem::deactivate_localization_mode,
           "Resume full SLAM mode.")
      .def("map_changed", &PySystem::map_changed,
           "Return whether the map changed since the previous call.")
      .def("reset", &PySystem::reset,
           "Reset the full system state.")
      .def("reset_active_map", &PySystem::reset_active_map,
           "Reset only the active map.")
      .def("shutdown", &PySystem::shutdown,
           "Request shutdown and wait for all internal threads to finish.")
      .def("is_shutdown", &PySystem::is_shutdown,
           "Return whether the system has already shut down.")
      .def("save_trajectory_tum", &PySystem::save_trajectory_tum,
           nb::arg("filename"),
           "Save the estimated trajectory in TUM format.")
      .def("save_keyframe_trajectory_tum", &PySystem::save_keyframe_trajectory_tum,
           nb::arg("filename"),
           "Save the keyframe trajectory in TUM format.")
      .def("save_trajectory_euroc", &PySystem::save_trajectory_euroc,
           nb::arg("filename"),
           "Save the estimated trajectory in EuRoC format.")
      .def("save_keyframe_trajectory_euroc", &PySystem::save_keyframe_trajectory_euroc,
           nb::arg("filename"),
           "Save the keyframe trajectory in EuRoC format.")
      .def("save_trajectory_kitti", &PySystem::save_trajectory_kitti,
           nb::arg("filename"),
           "Save the estimated trajectory in KITTI format.")
      .def("save_debug_data", &PySystem::save_debug_data,
           nb::arg("ini_idx"),
           "Save initialization debug data.")
      .def("get_tracking_state", &PySystem::get_tracking_state,
           "Return the current raw tracking state.")
      .def("get_time_from_imu_init", &PySystem::get_time_from_imu_init,
           "Return the IMU initialization time.")
      .def("is_lost", &PySystem::is_lost,
           "Return whether tracking is currently lost.")
      .def("is_finished", &PySystem::is_finished,
           "Return whether the system has finished processing.")
      .def("change_dataset", &PySystem::change_dataset,
           "Notify the system that the dataset has changed.")
      .def("get_image_scale", &PySystem::get_image_scale,
           "Return the current image scaling factor.");

  m.def("pose_identity", []() { return PoseSE3f{}; },
        "Return an identity pose.");
}