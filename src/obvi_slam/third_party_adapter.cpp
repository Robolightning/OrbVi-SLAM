#include "obvi_slam_bridge_runner.hpp"

#include <utility>

namespace obvi_bridge {

void install_obvi_slam_backend_from_third_party() {
  // Replace this placeholder with a real call into third_party/obvi_slam.
  //
  // The intended pattern is:
  //   - convert RunInput into the upstream ObVi-SLAM native types;
  //   - call the upstream offline pipeline function;
  //   - convert upstream results back into RunOutput;
  //   - set the native backend with set_native_backend(...).
  //
  // Once you know the exact function exported by third_party/obvi_slam,
  // change only this lambda and keep the rest of the bridge intact.
  set_native_backend(
      [](const RunInput &input) -> RunOutput {
        RunOutput out;
        out.success = false;
        out.error_message =
            "Third-party ObVi-SLAM adapter is present, but the real upstream "
            "call is not wired yet. "
            "Implement install_obvi_slam_backend_from_third_party() using the "
            "API exposed by third_party/obvi_slam.";
        out.warnings.push_back(describe_run_input(input));
        return out;
      });
}

}  // namespace obvi_bridge