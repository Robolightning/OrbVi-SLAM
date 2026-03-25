#include "obvi_slam_bridge_runner.hpp"

#include <sstream>
#include <stdexcept>

namespace obvi_bridge {
namespace {

std::mutex &backend_mutex() {
  static std::mutex m;
  return m;
}

NativeBackendFn &backend_slot() {
  static NativeBackendFn fn;
  return fn;
}

RunOutput placeholder_backend(const RunInput &) {
  RunOutput out;
  out.success = false;
  out.error_message =
      "ObVi-SLAM native backend is not installed yet. "
      "Wire third_party/obvi_slam in obvi_slam_third_party_adapter.cpp.";
  return out;
}

}  // namespace

void install_placeholder_backend() {
  std::lock_guard<std::mutex> lock(backend_mutex());
  backend_slot() = placeholder_backend;
}

void set_native_backend(NativeBackendFn fn) {
  std::lock_guard<std::mutex> lock(backend_mutex());
  if (fn) {
    backend_slot() = std::move(fn);
  } else {
    backend_slot() = placeholder_backend;
  }
}

std::string describe_run_input(const RunInput &input) {
  std::ostringstream oss;
  oss << "RunInput{"
      << "cameras=" << input.intrinsics_by_camera.size()
      << ", extrinsics=" << input.extrinsics_by_camera.size()
      << ", poses=" << input.robot_poses_by_frame.size()
      << ", frames=" << input.frames.size()
      << ", low_level_features=" << input.low_level_features.size()
      << ", has_ltm=" << (input.serialized_long_term_map_input.has_value() ? 1 : 0)
      << ", has_gt=" << (input.gt_trajectory.has_value() ? 1 : 0)
      << "}";
  return oss.str();
}

RunOutput run_obvi_slam_pipeline(const RunInput &input) {
  std::string why_not;
  if (!validate_run_input_shallow(input, why_not)) {
    RunOutput out;
    out.success = false;
    out.error_message = "Input validation failed: " + why_not;
    return out;
  }

  NativeBackendFn fn;
  {
    std::lock_guard<std::mutex> lock(backend_mutex());
    fn = backend_slot();
    if (!fn) {
      fn = placeholder_backend;
    }
  }

  try {
    RunOutput out = fn(input);
    if (!out.success && out.error_message.empty()) {
      out.error_message = "Backend returned failure without error_message.";
    }
    return out;
  } catch (const std::exception &e) {
    RunOutput out;
    out.success = false;
    out.error_message = std::string("Unhandled exception in backend: ") +
                        e.what();
    return out;
  } catch (...) {
    RunOutput out;
    out.success = false;
    out.error_message = "Unhandled non-standard exception in backend.";
    return out;
  }
}

}  // namespace obvi_bridge