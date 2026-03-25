#pragma once

#include "obvi_slam_bridge_types.hpp"

#include <functional>
#include <mutex>

namespace obvi_bridge {

using NativeBackendFn = std::function<RunOutput(const RunInput &)>;

// Install a concrete backend implementation. By default a placeholder backend
// is active and returns a failure result.
void set_native_backend(NativeBackendFn fn);

// Run the pipeline through the currently installed backend.
RunOutput run_obvi_slam_pipeline(const RunInput &input);

// Convenience helper for debugging from Python or CLI.
std::string describe_run_input(const RunInput &input);

// Explicitly install the placeholder backend.
void install_placeholder_backend();

}  // namespace obvi_bridge