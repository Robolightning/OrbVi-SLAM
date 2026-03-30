// Original code from ObVi‑SLAM, adapted for Windows compatibility

#include "util/timer.h"

#include <algorithm>
#include <string>

#include <glog/logging.h>

#ifdef _WIN32
#  define NOMINMAX           // prevent windows.h from defining min/max macros
#  include <windows.h>
#  include <intrin.h>        // for __rdtsc
#else
#  include <inttypes.h>
#  include <time.h>
#  include <unistd.h>
#endif

using std::max;
using std::string;

// -----------------------------------------------------------------------------
// RDTSC – Read Time‑Stamp Counter (x86/x64)
// -----------------------------------------------------------------------------
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
// MSVC x86/x64 intrinsic
uint64_t RDTSC() {
  return __rdtsc();
}
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
// GCC/Clang inline assembly
uint64_t RDTSC() {
  uint32_t hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return (static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));
}
#else
#  warning "RDTSC not implemented for this architecture; returning 0."
uint64_t RDTSC() { return 0; }
#endif

// -----------------------------------------------------------------------------
// GetWallTime – real‑time clock, seconds since Unix epoch (1970‑01‑01)
// -----------------------------------------------------------------------------
double GetWallTime() {
#ifdef _WIN32
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  // Convert FILETIME (100‑ns intervals since 1601‑01‑01) to Unix time.
  const uint64_t epoch_1601 = 116444736000000000ULL; // difference to 1970‑01‑01
  uint64_t ft_64 = (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
  const uint64_t unix_100ns = ft_64 - epoch_1601;
  return static_cast<double>(unix_100ns) * 1e-7; // 100 ns → seconds
#else
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

// -----------------------------------------------------------------------------
// GetMonotonicTime – monotonic (steady) clock, seconds
// -----------------------------------------------------------------------------
double GetMonotonicTime() {
#ifdef _WIN32
  static LARGE_INTEGER frequency = []() {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return freq;
  }();
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return static_cast<double>(counter.QuadPart) / static_cast<double>(frequency.QuadPart);
#else
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

// -----------------------------------------------------------------------------
// Sleep – pause the thread for a given number of seconds
// -----------------------------------------------------------------------------
void Sleep(double duration) {
  if (duration <= 0.0) return;
#ifdef _WIN32
  // Windows Sleep expects milliseconds, round up to at least 1 ms
  DWORD ms = static_cast<DWORD>(duration * 1000.0 + 0.5);
  if (ms == 0) ms = 1;
  ::Sleep(ms);
#else
  const useconds_t usec = static_cast<useconds_t>(duration * 1e6);
  usleep(usec);
#endif
}

// -----------------------------------------------------------------------------
// RateLoop
// -----------------------------------------------------------------------------
RateLoop::RateLoop(double rate) :
    t_last_run_(0.0), delay_interval_(1.0 / rate) { }

void RateLoop::Sleep() {
  const double t_now = GetMonotonicTime();
  const double sleep_duration = max(0.0, delay_interval_ + t_last_run_ - t_now);
  ::Sleep(sleep_duration);
  t_last_run_ = t_now + sleep_duration;
}

// -----------------------------------------------------------------------------
// FunctionTimer
// -----------------------------------------------------------------------------
FunctionTimer::FunctionTimer(const char* name) :
    name_(name), t_start_(GetMonotonicTime()), t_lap_start_(t_start_) {}

FunctionTimer::~FunctionTimer() {
  const double t_stop = GetMonotonicTime();
  LOG(INFO) << name_ << ": " << (1.0E3 * (t_stop - t_start_)) << " ms";
}

void FunctionTimer::Lap(int id) {
  const double t_now = GetMonotonicTime();
  LOG(INFO) << name_ << "(" << id << "): " << (1.0E3 * (t_now - t_lap_start_)) << " ms";
  t_lap_start_ = t_now;
}

// -----------------------------------------------------------------------------
// CumulativeFunctionTimer
// -----------------------------------------------------------------------------
CumulativeFunctionTimer::Invocation::Invocation(CumulativeFunctionTimer* cumulative_timer) :
    t_start_(GetMonotonicTime()), cumulative_timer_(cumulative_timer) {}

CumulativeFunctionTimer::Invocation::~Invocation() {
  const double t_duration = GetMonotonicTime() - t_start_;
  cumulative_timer_->total_invocations_++;
  cumulative_timer_->total_run_time_ += t_duration;
}

CumulativeFunctionTimer::CumulativeFunctionTimer(const char* name) :
    name_(name), total_run_time_(0.0), total_invocations_(0) {}

CumulativeFunctionTimer::~CumulativeFunctionTimer() {
  const double mean_run_time = total_run_time_ / static_cast<double>(total_invocations_);
  LOG(INFO) << "Run-time stats for " << name_ << " : mean run time = " << (1.0E3 * mean_run_time)
            << " ms, invocations = " << total_invocations_;
}

// -----------------------------------------------------------------------------
// RateChecker
// -----------------------------------------------------------------------------
RateChecker::Invocation::Invocation(RateChecker* rate_checker) :
    t_start_(GetMonotonicTime()), rate_checker_(rate_checker) {
  rate_checker_->total_invocations_++;
  if (rate_checker_->t_last_run_ > 0.0) {
    rate_checker_->total_intervals_ += (t_start_ - rate_checker_->t_last_run_);
  }
  rate_checker_->t_last_run_ = t_start_;
}

RateChecker::Invocation::~Invocation() {
  const double t_duration = GetMonotonicTime() - t_start_;
  rate_checker_->total_run_time_ += t_duration;
}

RateChecker::RateChecker(const char* name) :
    name_(name),
    t_last_run_(0.0),
    total_intervals_(0),
    total_run_time_(0.0),
    total_invocations_(0) {}

RateChecker::~RateChecker() {
  const double mean_run_time = total_run_time_ / static_cast<double>(total_invocations_);
  const double mean_interval = total_intervals_ / static_cast<double>(total_invocations_);
  LOG(INFO) << "Run-time stats for " << name_ << " : mean run time = " << (1.0E3 * mean_run_time)
            << " ms, invocations = " << total_invocations_
            << ", mean interval = " << (1.0E3 * mean_interval) << " ms";
}