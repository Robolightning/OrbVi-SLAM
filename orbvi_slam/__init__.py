import cv2
import numpy as np
import torch

from ._orb_slam3 import System, Sensor, IMUPoint, KeyFrame
from ._obvi_slam import (
    ObViSlamAdapter,
    RawBoundingBox,
    BbCornerPair,
    PixelCoord,
    EllipsoidState,
    Pose3D
)


class OrbViSlam:
    """
    High-level class integrating ORB-SLAM3, ObVi-SLAM and YOLO detection,
    optionally with IMU data.

    Args:
        orb_vocab_path (str): Path to ORB vocabulary file.
        orb_config_path (str): Path to ORB-SLAM3 configuration file (YAML).
        obvi_config_path (str): Path to ObVi-SLAM configuration file (YAML).
        sensor (int): ORB-SLAM3 sensor type (e.g., Sensor.MONOCULAR, Sensor.IMU_MONOCULAR).
        detector (optional): Object detector with a `detect` method.
            If None, YOLOv5 (via torch.hub) is used as a fallback.
        camera_id (int): Camera identifier (default 0).
        use_viewer (bool): Whether to show ORB-SLAM3 viewer (default False).
    """
    def __init__(self, orb_vocab_path, orb_config_path, obvi_config_path,
                 sensor=Sensor.MONOCULAR, detector=None, camera_id=0, use_viewer=False):
        self.camera_id = camera_id
        self.frame_counter = 0
        self.last_keyframe_timestamp = None

        # ORB-SLAM3
        self.orb = System(orb_vocab_path, orb_config_path, sensor, use_viewer)
        self.sensor = sensor

        # ObVi-SLAM adapter
        self.adapter = ObViSlamAdapter(obvi_config_path)

        # YOLO detector
        self.detector = detector
        if self.detector is None:
            # Load YOLOv5 as placeholder (replace with YOLOv26)
            self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            self.detector.conf = 0.5

    def _process_orb(self, image, timestamp, imu_meas=None):
        """
        Run ORB-SLAM3 on the image with optional IMU data.
        Returns pose matrix and keyframe data if new keyframe created.
        """
        if imu_meas is None:
            pose_mat = self.orb.track_monocular(image, timestamp)
        else:
            # Convert imu_meas (list of tuples) to list of IMUPoint
            imu_points = []
            for t, gx, gy, gz, ax, ay, az in imu_meas:
                p = IMUPoint(ax, ay, az, gx, gy, gz, t)   # note argument order: ax,ay,az,gx,gy,gz,timestamp
                imu_points.append(p)
            pose_mat = self.orb.track_monocular(image, timestamp, imu_points)

        kf = self.orb.get_last_keyframe()
        if kf is not None and kf.get_timestamp() == timestamp:
            # New keyframe
            # Convert pose from float32 Fortran-order to float64 C-order as required by adapter
            kf_pose = np.asarray(kf.get_pose(), dtype=np.float64, order='C')
            keypoints = np.array(kf.get_keypoints(), dtype=np.float64)
            descriptors = np.asarray(kf.get_descriptors(), dtype=np.uint8)
            if descriptors.size == 0:
                descriptors = descriptors.reshape(0, 32)
            elif descriptors.ndim == 1:
                descriptors = descriptors.reshape(-1, 32)
            # Image size: adapter expects (width, height)
            width = image.shape[1] if len(image.shape) > 1 else image.shape[0]  # handle 1D fallback
            height = image.shape[0]
            img_size = (width, height)
            return pose_mat, (kf_pose, keypoints, descriptors, img_size)
        else:
            return pose_mat, None

    def _process_yolo(self, image):
        """Run YOLO detection on the image. Returns list of RawBoundingBox objects."""
        results = self.detector(image)
        detections = []
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            bb = RawBoundingBox()
            # Create BbCornerPair and assign pixel coordinates
            bb.pixel_corner_locations = BbCornerPair()
            # Use PixelCoord if available; otherwise fall back to tuple (if conversion works)
            # If PixelCoord is not exported, replace with (x1, y1) and (x2, y2) directly.
            try:
                bb.pixel_corner_locations.first = PixelCoord(x1, y1)
                bb.pixel_corner_locations.second = PixelCoord(x2, y2)
            except NameError:
                # PixelCoord not defined – use tuples (may work if bindings support tuple conversion)
                bb.pixel_corner_locations.first = (x1, y1)
                bb.pixel_corner_locations.second = (x2, y2)
            bb.semantic_class = str(int(cls))
            bb.detection_confidence = conf
            detections.append(bb)
        return detections

    def process_frame(self, image, timestamp, imu_measurements=None):
        """
        Process a single frame: run ORB-SLAM3 and YOLO, feed data to ObVi-SLAM.

        Args:
            image (np.ndarray): BGR or grayscale image.
            timestamp (float): Frame timestamp (seconds).
            imu_measurements (list of tuple, optional): List of IMU measurements
                between previous frame and this one. Each tuple: (t, gx, gy, gz, ax, ay, az).
        """
        # Convert to grayscale for ORB-SLAM3
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # ORB-SLAM3
        pose_mat, keyframe_data = self._process_orb(gray, timestamp, imu_measurements)

        # Add keyframe to adapter if new keyframe was created
        if keyframe_data is not None:
            kf_pose, keypoints, descriptors, img_size = keyframe_data
            self.adapter.add_keyframe(
                frame_id=self.frame_counter,
                pose_matrix=kf_pose,
                keypoints=keypoints,
                descriptors=descriptors,
                image_sizes={self.camera_id: img_size}
            )

        # YOLO detection
        detections = self._process_yolo(image)

        # Add detections for this frame (even if not a keyframe)
        self.adapter.add_detections(
            frame_id=self.frame_counter,
            detections={self.camera_id: detections}
        )

        self.frame_counter += 1

    def finalize(self):
        """Run final optimization after processing all frames."""
        if not self.adapter.optimize():
            raise RuntimeError("ObVi-SLAM optimization failed")

    def get_trajectory(self):
        """Return refined camera trajectory (dict frame_id -> Pose3D)."""
        return self.adapter.get_optimized_trajectory()

    def get_object_map(self):
        """Return refined object map (dict object_id -> EllipsoidState)."""
        return self.adapter.get_object_map()