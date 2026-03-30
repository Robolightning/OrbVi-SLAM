#!/usr/bin/env python3
"""
Example of running OrbVi-SLAM on a video sequence without IMU.
"""

import cv2
import numpy as np
import orbvi_slam as ovs

def main():
    # Paths to configuration files
    vocab_path = "data/orb_vocab.txt"
    orb_config_path = "data/orb_config.yaml"
    obvi_config_path = "data/obvi_config.yaml"
    video_path = "data/sequence.mp4"

    # Initialize the integrated system
    slam = ovs.OrbViSlam(
        orb_vocab_path=vocab_path,
        orb_config_path=orb_config_path,
        obvi_config_path=obvi_config_path,
        sensor=ovs.Sensor.MONOCULAR,  # standard monocular mode
        use_viewer=False
    )

    # Set camera intrinsics and extrinsics (must be done before optimization)
    # Example for a monocular camera (replace with actual calibration)
    fx, fy, cx, cy = 718.856, 718.856, 607.1928, 185.2157  # KITTI example
    intrinsics = {
        0: np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    }
    extrinsics = {
        0: np.eye(4, dtype=np.float64)  # camera is the robot body frame
    }
    slam.set_camera_intrinsics(intrinsics)
    slam.set_camera_extrinsics(extrinsics)

    # Process video frames
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / 30.0  # dummy timestamps, replace with actual if available
        slam.process_frame(frame, timestamp)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")
    cap.release()
    print(f"Total frames processed: {frame_idx}")

    # Run final optimization
    print("Running final optimization...")
    slam.finalize()
    print("Optimization finished.")

    # Retrieve results
    traj = slam.get_trajectory()
    obj_map = slam.get_object_map()

    # Save trajectory (TUM format: timestamp tx ty tz qx qy qz qw)
    with open("trajectory.txt", "w") as f:
        for fid, pose in sorted(traj.items()):
            t = pose.translation
            q = pose.rotation_quaternion
            # Assume timestamp = fid * 0.033 (if fps=30)
            timestamp = fid * 0.033
            f.write(f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    # Save object map
    with open("objects.txt", "w") as f:
        for oid, (cls, state) in obj_map.items():
            c = state.pose.translation
            d = state.dimensions
            f.write(f"{oid} {cls} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                    f"{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}\n")

    print("Done. Results saved to trajectory.txt and objects.txt")

if __name__ == "__main__":
    main()