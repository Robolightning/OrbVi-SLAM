#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <opencv2/core.hpp>

#include <array>
#include <stdexcept>

#include "System.h"
#include "KeyFrame.h"
#include "ImuTypes.h"

namespace nb = nanobind;


static cv::Mat ndarray_to_gray_mat(nb::ndarray<uint8_t, nb::shape<-1, -1>, nb::c_contig> img) {
    if (img.ndim() != 2) {
        throw std::runtime_error("ORB-SLAM3 expects a 2D grayscale image (H x W).");
    }
    return cv::Mat(static_cast<int>(img.shape(0)),
                   static_cast<int>(img.shape(1)),
                   CV_8UC1,
                   const_cast<uint8_t*>(img.data()));
}


NB_MODULE(_orb_slam3, m) {
    m.doc() = "Python bindings for ORB-SLAM3 (ROS-free version)";

    // Sensor enum
    nb::enum_<ORB_SLAM3::System::eSensor>(m, "Sensor", "Sensor type enumeration")
        .value("MONOCULAR", ORB_SLAM3::System::MONOCULAR, "Monocular camera")
        .value("STEREO", ORB_SLAM3::System::STEREO, "Stereo camera")
        .value("RGBD", ORB_SLAM3::System::RGBD, "RGB-D camera")
        .value("IMU_MONOCULAR", ORB_SLAM3::System::IMU_MONOCULAR, "Monocular with IMU")
        .value("IMU_STEREO", ORB_SLAM3::System::IMU_STEREO, "Stereo with IMU")
        .value("IMU_RGBD", ORB_SLAM3::System::IMU_RGBD, "RGB-D with IMU")
        .export_values();

    // IMU measurement point
    nb::class_<ORB_SLAM3::IMU::Point>(m, "IMUPoint", "IMU measurement")
        .def(nb::init<float, float, float, float, float, float, double>(),
            nb::arg("ax"), nb::arg("ay"), nb::arg("az"),
            nb::arg("gx"), nb::arg("gy"), nb::arg("gz"),
            nb::arg("timestamp"),
            "Construct from acceleration (ax,ay,az), angular velocity (gx,gy,gz), and timestamp")
        .def_rw("t", &ORB_SLAM3::IMU::Point::t, "Timestamp (seconds)")
        .def_rw("w", &ORB_SLAM3::IMU::Point::w, "Angular velocity (gx, gy, gz) [rad/s]")
        .def_rw("a", &ORB_SLAM3::IMU::Point::a, "Linear acceleration (ax, ay, az) [m/s^2]");

    // KeyFrame
    nb::class_<ORB_SLAM3::KeyFrame>(m, "KeyFrame", "ORB-SLAM3 keyframe")
        .def("get_pose", [](ORB_SLAM3::KeyFrame& kf) {
            Eigen::Matrix4f mat = kf.GetPose().matrix();
            return nb::ndarray<float, nb::shape<4, 4>, nb::f_contig>(mat.data(), { 4,4 });
        }, R"doc(
        Get the camera pose as a 4x4 transformation matrix.

        Returns:
            np.ndarray[float32, shape=(4,4)]: Camera pose in world coordinate system.
        )doc")
        .def("get_timestamp", [](const ORB_SLAM3::KeyFrame& kf) { return kf.mTimeStamp;
        }, R"doc(
        Get the timestamp of the keyframe.

        Returns:
            float: Timestamp in seconds.
        )doc")
        .def("get_keypoints", [](const ORB_SLAM3::KeyFrame &kf) -> nb::list {
            nb::list keypoints;
            for (const auto &kp : kf.mvKeysUn) {
                keypoints.append(nb::make_tuple(kp.pt.x, kp.pt.y));
            }
            return keypoints;
        }, R"doc(
        Get the keypoints as a list of (x, y) tuples.

        Returns:
            list[tuple[float, float]]: List of keypoint coordinates.
        )doc")
        .def("get_descriptors", [](const ORB_SLAM3::KeyFrame &kf) {
            std::vector<std::vector<uint8_t>> out;
            out.reserve(kf.mDescriptors.rows);

            for (int i = 0; i < kf.mDescriptors.rows; ++i) {
                const uint8_t* row = kf.mDescriptors.ptr<uint8_t>(i);
                out.emplace_back(row, row + kf.mDescriptors.cols);
            }
            return out;
        }, R"doc(
        Get the ORB descriptors as a list of 32-byte rows.
        )doc");

    // System
    nb::class_<ORB_SLAM3::System>(m, "System", "ORB-SLAM3 system main interface")
        .def(nb::init<const std::string&, const std::string&, ORB_SLAM3::System::eSensor, bool>(),
             nb::arg("voc_file"), nb::arg("settings_file"), nb::arg("sensor"),
             nb::arg("use_viewer") = true,
             R"doc(
             Initialize ORB-SLAM3 system.

             Args:
                 voc_file (str): Path to ORB vocabulary file.
                 settings_file (str): Path to camera settings YAML file.
                 sensor (Sensor): Sensor type (e.g., Sensor.MONOCULAR).
                 use_viewer (bool): Whether to enable the GUI viewer (default: True).
             )doc")
        .def("track_monocular",
        [](ORB_SLAM3::System &sys,
            nb::ndarray<uint8_t, nb::shape<-1, -1>, nb::c_contig> im,
            double timestamp) -> Eigen::Matrix4d {
            cv::Mat mat = ndarray_to_gray_mat(im);
            Sophus::SE3f pose = sys.TrackMonocular(mat, timestamp);
            return pose.matrix().cast<double>();
        },
        nb::arg("image"), nb::arg("timestamp"),
        R"doc(
        Process a monocular image frame without IMU data.

        Args:
            image (np.ndarray): Grayscale image (CV_8U) or RGB (CV_8UC3). RGB is converted to grayscale.
            timestamp (float): Timestamp in seconds.

        Returns:
            np.ndarray[float32, shape=(4,4)]: Camera pose (empty if tracking fails).
        )doc")
        .def("track_monocular",
            [](ORB_SLAM3::System &sys,
            nb::ndarray<uint8_t, nb::shape<-1, -1>, nb::c_contig> im,
            double timestamp,
            const std::vector<ORB_SLAM3::IMU::Point>& imu_meas) -> Eigen::Matrix4d {
            cv::Mat mat = ndarray_to_gray_mat(im);
            Sophus::SE3f pose = sys.TrackMonocular(mat, timestamp, imu_meas);
            return pose.matrix().cast<double>();
        },
        nb::arg("image"), nb::arg("timestamp"), nb::arg("imu_measurements"),
        R"doc(
        Process a monocular image frame with IMU data.

        Args:
            image (np.ndarray): Grayscale image (CV_8U) or RGB (CV_8UC3). RGB is converted to grayscale.
            timestamp (float): Timestamp in seconds.
            imu_measurements (list[IMUPoint]): List of IMU measurements between previous and current frame.

        Returns:
            np.ndarray[float32, shape=(4,4)]: Camera pose (empty if tracking fails).
        )doc")
        .def("get_last_keyframe", &ORB_SLAM3::System::GetLastKeyFrame,
             nb::rv_policy::reference_internal,
             R"doc(Get the last inserted keyframe.)doc")
        .def("shutdown", &ORB_SLAM3::System::Shutdown,
             R"doc(Shutdown all system threads.)doc")
        .def("reset", &ORB_SLAM3::System::Reset,
             R"doc(Reset the system (clear atlas and maps).)doc");
}