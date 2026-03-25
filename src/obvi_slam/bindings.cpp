#include "obvi_slam_bridge_runner.hpp"
#include "obvi_slam_bridge_types.hpp"

#include <Python.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace obvi_bridge;

namespace {

std::vector<std::uint8_t> bytes_to_vector(nb::handle obj) {
  if (!PyBytes_Check(obj.ptr())) {
    throw nb::type_error("Expected Python bytes object.");
  }

  char *buffer = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(obj.ptr(), &buffer, &size) != 0) {
    throw nb::python_error();
  }

  return std::vector<std::uint8_t>(buffer, buffer + size);
}

nb::bytes vector_to_bytes(const std::vector<std::uint8_t> &data) {
  return nb::bytes(reinterpret_cast<const char *>(data.data()), data.size());
}

ImageBuffer make_image_buffer_from_bytes(int width,
                                         int height,
                                         int channels,
                                         PixelFormat pixel_format,
                                         const std::string &encoding,
                                         nb::handle data_bytes) {
  ImageBuffer img;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.pixel_format = pixel_format;
  img.encoding = encoding;
  img.data = bytes_to_vector(data_bytes);
  return img;
}

}  // namespace

NB_MODULE(obvi_slam_bridge, m) {
  m.doc() = "Python bridge for ObVi-SLAM without ROS";

  install_placeholder_backend();
  install_obvi_slam_backend_from_third_party();

  nb::enum_<PixelFormat>(m, "PixelFormat")
      .value("GRAY8", PixelFormat::GRAY8)
      .value("RGB8", PixelFormat::RGB8)
      .value("BGR8", PixelFormat::BGR8)
      .value("RGBA8", PixelFormat::RGBA8)
      .value("BGRA8", PixelFormat::BGRA8)
      .export_values();

  nb::class_<ImageBuffer>(m, "ImageBuffer")
      .def(nb::init<>())
      .def_rw("width", &ImageBuffer::width)
      .def_rw("height", &ImageBuffer::height)
      .def_rw("channels", &ImageBuffer::channels)
      .def_rw("pixel_format", &ImageBuffer::pixel_format)
      .def_rw("encoding", &ImageBuffer::encoding)
      .def_rw("data", &ImageBuffer::data)
      .def("empty", &ImageBuffer::empty)
      .def("size_bytes", &ImageBuffer::size_bytes);

  nb::class_<CameraIntrinsics>(m, "CameraIntrinsics")
      .def(nb::init<>())
      .def_rw("camera_id", &CameraIntrinsics::camera_id)
      .def_rw("width", &CameraIntrinsics::width)
      .def_rw("height", &CameraIntrinsics::height)
      .def_rw("fx", &CameraIntrinsics::fx)
      .def_rw("fy", &CameraIntrinsics::fy)
      .def_rw("cx", &CameraIntrinsics::cx)
      .def_rw("cy", &CameraIntrinsics::cy)
      .def_rw("distortion_model", &CameraIntrinsics::distortion_model)
      .def_rw("distortion_coeffs", &CameraIntrinsics::distortion_coeffs)
      .def("is_valid", &CameraIntrinsics::is_valid);

  nb::class_<CameraExtrinsics>(m, "CameraExtrinsics")
      .def(nb::init<>())
      .def_rw("camera_id", &CameraExtrinsics::camera_id)
      .def_rw("T_body_cam", &CameraExtrinsics::T_body_cam)
      .def("is_valid", &CameraExtrinsics::is_valid);

  nb::class_<BoundingBoxDetection>(m, "BoundingBoxDetection")
      .def(nb::init<>())
      .def_rw("camera_id", &BoundingBoxDetection::camera_id)
      .def_rw("semantic_class", &BoundingBoxDetection::semantic_class)
      .def_rw("confidence", &BoundingBoxDetection::confidence)
      .def_rw("min_x", &BoundingBoxDetection::min_x)
      .def_rw("min_y", &BoundingBoxDetection::min_y)
      .def_rw("max_x", &BoundingBoxDetection::max_x)
      .def_rw("max_y", &BoundingBoxDetection::max_y)
      .def("is_valid", &BoundingBoxDetection::is_valid);

  nb::class_<Pose3D>(m, "Pose3D")
      .def(nb::init<>())
      .def_rw("frame_id", &Pose3D::frame_id)
      .def_rw("timestamp_ns", &Pose3D::timestamp_ns)
      .def_rw("t", &Pose3D::t)
      .def_rw("q", &Pose3D::q)
      .def("is_valid", &Pose3D::is_valid);

  nb::class_<FeatureObservation>(m, "FeatureObservation")
      .def(nb::init<>())
      .def_rw("frame_id", &FeatureObservation::frame_id)
      .def_rw("camera_id", &FeatureObservation::camera_id)
      .def_rw("x", &FeatureObservation::x)
      .def_rw("y", &FeatureObservation::y)
      .def_rw("uncertainty", &FeatureObservation::uncertainty)
      .def("is_valid", &FeatureObservation::is_valid);

  nb::class_<LowLevelFeatureTrack>(m, "LowLevelFeatureTrack")
      .def(nb::init<>())
      .def_rw("feature_id", &LowLevelFeatureTrack::feature_id)
      .def_rw("position_w", &LowLevelFeatureTrack::position_w)
      .def_rw("observations", &LowLevelFeatureTrack::observations)
      .def("is_valid", &LowLevelFeatureTrack::is_valid);

  nb::class_<FrameInput>(m, "FrameInput")
      .def(nb::init<>())
      .def_rw("frame_id", &FrameInput::frame_id)
      .def_rw("timestamp_ns", &FrameInput::timestamp_ns)
      .def_rw("images", &FrameInput::images)
      .def_rw("detections", &FrameInput::detections)
      .def("is_valid", &FrameInput::is_valid);

  nb::class_<RunConfig>(m, "RunConfig")
      .def(nb::init<>())
      .def_rw("params_config_file", &RunConfig::params_config_file)
      .def_rw("logs_directory", &RunConfig::logs_directory)
      .def_rw("output_checkpoints_dir", &RunConfig::output_checkpoints_dir)
      .def_rw("debug_images_output_directory",
              &RunConfig::debug_images_output_directory)
      .def_rw("ltm_opt_jacobian_info_directory",
              &RunConfig::ltm_opt_jacobian_info_directory)
      .def_rw("visual_feature_results_file",
              &RunConfig::visual_feature_results_file)
      .def_rw("bb_associations_out_file", &RunConfig::bb_associations_out_file)
      .def_rw("ellipsoids_results_file", &RunConfig::ellipsoids_results_file)
      .def_rw("robot_poses_results_file", &RunConfig::robot_poses_results_file)
      .def_rw("long_term_map_output_file", &RunConfig::long_term_map_output_file)
      .def_rw("enable_visualization", &RunConfig::enable_visualization)
      .def_rw("write_files", &RunConfig::write_files)
      .def_rw("disable_log_to_stderr", &RunConfig::disable_log_to_stderr);

  nb::class_<RunInput>(m, "RunInput")
      .def(nb::init<>())
      .def_rw("config", &RunInput::config)
      .def_rw("intrinsics_by_camera", &RunInput::intrinsics_by_camera)
      .def_rw("extrinsics_by_camera", &RunInput::extrinsics_by_camera)
      .def_rw("robot_poses_by_frame", &RunInput::robot_poses_by_frame)
      .def_rw("frames", &RunInput::frames)
      .def_rw("low_level_features", &RunInput::low_level_features)
      .def_rw("serialized_long_term_map_input",
              &RunInput::serialized_long_term_map_input)
      .def_rw("gt_trajectory", &RunInput::gt_trajectory);

  nb::class_<RobotPoseResult>(m, "RobotPoseResult")
      .def(nb::init<>())
      .def_rw("frame_id", &RobotPoseResult::frame_id)
      .def_rw("pose", &RobotPoseResult::pose);

  nb::class_<EllipsoidResult>(m, "EllipsoidResult")
      .def(nb::init<>())
      .def_rw("object_id", &EllipsoidResult::object_id)
      .def_rw("semantic_class", &EllipsoidResult::semantic_class)
      .def_rw("center", &EllipsoidResult::center)
      .def_rw("radii", &EllipsoidResult::radii)
      .def_rw("q", &EllipsoidResult::q)
      .def_rw("covariance_flat", &EllipsoidResult::covariance_flat);

  nb::class_<BoundingBoxAssociationResult>(m, "BoundingBoxAssociationResult")
      .def(nb::init<>())
      .def_rw("frame_id", &BoundingBoxAssociationResult::frame_id)
      .def_rw("camera_id", &BoundingBoxAssociationResult::camera_id)
      .def_rw("object_id", &BoundingBoxAssociationResult::object_id)
      .def_rw("detection", &BoundingBoxAssociationResult::detection);

  nb::class_<VisualFeatureResult>(m, "VisualFeatureResult")
      .def(nb::init<>())
      .def_rw("feature_id", &VisualFeatureResult::feature_id)
      .def_rw("position_w", &VisualFeatureResult::position_w);

  nb::class_<RunOutput>(m, "RunOutput")
      .def(nb::init<>())
      .def_rw("success", &RunOutput::success)
      .def_rw("error_message", &RunOutput::error_message)
      .def_rw("warnings", &RunOutput::warnings)
      .def_rw("optimized_robot_poses", &RunOutput::optimized_robot_poses)
      .def_rw("ellipsoids", &RunOutput::ellipsoids)
      .def_rw("associations", &RunOutput::associations)
      .def_rw("visual_features", &RunOutput::visual_features)
      .def_rw("serialized_long_term_map", &RunOutput::serialized_long_term_map)
      .def_rw("serialized_pose_graph", &RunOutput::serialized_pose_graph);

  m.def("make_image_buffer_from_bytes", &make_image_buffer_from_bytes,
        nb::arg("width"), nb::arg("height"), nb::arg("channels"),
        nb::arg("pixel_format"), nb::arg("encoding"), nb::arg("data_bytes"));

  m.def("image_buffer_to_bytes", &vector_to_bytes, nb::arg("image").noconvert(),
        "Convert ImageBuffer.data to Python bytes.");

  m.def("describe_run_input", &describe_run_input);
  m.def("run_obvi_slam_pipeline", &run_obvi_slam_pipeline);
}