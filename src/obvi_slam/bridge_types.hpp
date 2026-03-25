#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace obvi_bridge {

using CameraId = std::int64_t;
using FrameId = std::int64_t;
using FeatureId = std::int64_t;
using ObjectId = std::int64_t;
using TimestampNs = std::int64_t;

enum class PixelFormat : std::int32_t {
  GRAY8 = 0,
  RGB8 = 1,
  BGR8 = 2,
  RGBA8 = 3,
  BGRA8 = 4
};

inline std::array<double, 16> identity4x4() {
  return {1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0};
}

struct ImageBuffer {
  int width = 0;
  int height = 0;
  int channels = 0;
  PixelFormat pixel_format = PixelFormat::BGR8;
  std::string encoding = "bgr8";
  std::vector<std::uint8_t> data;

  bool empty() const { return width <= 0 || height <= 0 || data.empty(); }
  std::size_t size_bytes() const { return data.size(); }
};

struct CameraIntrinsics {
  CameraId camera_id = -1;
  int width = 0;
  int height = 0;

  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;

  std::string distortion_model = "plumb_bob";
  std::vector<double> distortion_coeffs;

  bool is_valid() const {
    return camera_id >= 0 && width > 0 && height > 0 && fx != 0.0 &&
           fy != 0.0;
  }
};

struct CameraExtrinsics {
  CameraId camera_id = -1;

  // Row-major 4x4 transform from body/base frame to camera frame.
  std::array<double, 16> T_body_cam = identity4x4();

  bool is_valid() const { return camera_id >= 0; }
};

struct BoundingBoxDetection {
  CameraId camera_id = -1;
  int semantic_class = -1;
  double confidence = 0.0;

  double min_x = 0.0;
  double min_y = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;

  bool is_valid() const {
    return camera_id >= 0 && max_x >= min_x && max_y >= min_y;
  }
};

struct Pose3D {
  FrameId frame_id = -1;
  TimestampNs timestamp_ns = 0;

  // Translation in meters.
  std::array<double, 3> t = {0.0, 0.0, 0.0};

  // Quaternion in w, x, y, z order.
  std::array<double, 4> q = {1.0, 0.0, 0.0, 0.0};

  bool is_valid() const { return frame_id >= 0; }
};

struct FeatureObservation {
  FrameId frame_id = -1;
  CameraId camera_id = -1;
  double x = 0.0;
  double y = 0.0;
  std::optional<double> uncertainty;

  bool is_valid() const { return frame_id >= 0 && camera_id >= 0; }
};

struct LowLevelFeatureTrack {
  FeatureId feature_id = -1;
  std::array<double, 3> position_w = {0.0, 0.0, 0.0};
  std::vector<FeatureObservation> observations;

  bool is_valid() const { return feature_id >= 0; }
};

struct FrameInput {
  FrameId frame_id = -1;
  TimestampNs timestamp_ns = 0;

  std::unordered_map<CameraId, ImageBuffer> images;
  std::unordered_map<CameraId, std::vector<BoundingBoxDetection>> detections;

  bool is_valid() const { return frame_id >= 0; }
};

struct RunConfig {
  // Keep this because the upstream ObVi-SLAM config is large. You can still
  // switch to an in-memory config DTO later.
  std::string params_config_file;

  // Optional file outputs; can be left empty when you only want the in-memory
  // result.
  std::string logs_directory;
  std::string output_checkpoints_dir;
  std::string debug_images_output_directory;
  std::string ltm_opt_jacobian_info_directory;
  std::string visual_feature_results_file;
  std::string bb_associations_out_file;
  std::string ellipsoids_results_file;
  std::string robot_poses_results_file;
  std::string long_term_map_output_file;

  bool enable_visualization = false;
  bool write_files = false;
  bool disable_log_to_stderr = false;
};

struct RunInput {
  RunConfig config;

  std::unordered_map<CameraId, CameraIntrinsics> intrinsics_by_camera;
  std::unordered_map<CameraId, CameraExtrinsics> extrinsics_by_camera;

  // Initial trajectory estimates.
  std::unordered_map<FrameId, Pose3D> robot_poses_by_frame;

  // Full batch input, already grouped by frame.
  std::vector<FrameInput> frames;

  // Optional low-level feature tracks, if the upstream pipeline consumes them.
  std::unordered_map<FeatureId, LowLevelFeatureTrack> low_level_features;

  // Optional initial long-term map payload. Keep it serialized at this layer,
  // so you do not leak upstream map classes into Python.
  std::optional<std::string> serialized_long_term_map_input;

  // Optional ground-truth trajectory for evaluation.
  std::optional<std::vector<Pose3D>> gt_trajectory;
};

struct RobotPoseResult {
  FrameId frame_id = -1;
  Pose3D pose;
};

struct EllipsoidResult {
  ObjectId object_id = -1;
  std::string semantic_class;

  // Center (x, y, z)
  std::array<double, 3> center = {0.0, 0.0, 0.0};

  // Principal radii (a, b, c)
  std::array<double, 3> radii = {0.0, 0.0, 0.0};

  // Orientation quaternion w, x, y, z.
  std::array<double, 4> q = {1.0, 0.0, 0.0, 0.0};

  // Optional covariance flattened row-major. Keep it generic.
  std::vector<double> covariance_flat;
};

struct BoundingBoxAssociationResult {
  FrameId frame_id = -1;
  CameraId camera_id = -1;
  ObjectId object_id = -1;
  BoundingBoxDetection detection;
};

struct VisualFeatureResult {
  FeatureId feature_id = -1;
  std::array<double, 3> position_w = {0.0, 0.0, 0.0};
};

struct RunOutput {
  bool success = false;
  std::string error_message;
  std::vector<std::string> warnings;

  std::vector<RobotPoseResult> optimized_robot_poses;
  std::vector<EllipsoidResult> ellipsoids;
  std::vector<BoundingBoxAssociationResult> associations;
  std::vector<VisualFeatureResult> visual_features;

  // Optional serialized artifacts to return to Python if needed.
  std::optional<std::string> serialized_long_term_map;
  std::optional<std::string> serialized_pose_graph;
};

inline bool validate_image_buffer(const ImageBuffer &img) {
  return img.width > 0 && img.height > 0 && !img.data.empty();
}

inline bool validate_run_input_shallow(const RunInput &input,
                                       std::string &why_not) {
  if (input.intrinsics_by_camera.empty()) {
    why_not = "intrinsics_by_camera is empty";
    return false;
  }
  if (input.extrinsics_by_camera.empty()) {
    why_not = "extrinsics_by_camera is empty";
    return false;
  }
  if (input.frames.empty()) {
    why_not = "frames is empty";
    return false;
  }
  for (const auto &kv : input.intrinsics_by_camera) {
    if (!kv.second.is_valid()) {
      why_not = "invalid camera intrinsics for camera_id=" +
                std::to_string(kv.first);
      return false;
    }
  }
  for (const auto &kv : input.extrinsics_by_camera) {
    if (!kv.second.is_valid()) {
      why_not = "invalid camera extrinsics for camera_id=" +
                std::to_string(kv.first);
      return false;
    }
  }
  for (const auto &frame : input.frames) {
    if (!frame.is_valid()) {
      why_not = "invalid frame input";
      return false;
    }
    if (frame.images.empty()) {
      why_not = "frame_id=" + std::to_string(frame.frame_id) +
                " has no images";
      return false;
    }
    for (const auto &cam_img : frame.images) {
      if (!validate_image_buffer(cam_img.second)) {
        why_not = "invalid image buffer for frame_id=" +
                  std::to_string(frame.frame_id) +
                  ", camera_id=" + std::to_string(cam_img.first);
        return false;
      }
    }
  }
  return true;
}

}  // namespace obvi_bridge