#!/usr/bin/env python3
"""
Example of running OrbVi-SLAM on a video sequence with IMU data.
Assumes IMU data is stored in a CSV file with columns: timestamp, gx, gy, gz, ax, ay, az.
"""

import cv2
import numpy as np
import orbvi_slam as ovs
import csv
import time

def read_imu_data(imu_file_path):
    """
    Read IMU data from a CSV file.
    Returns a list of tuples (timestamp, gx, gy, gz, ax, ay, az).
    """
    imu_data = []
    with open(imu_file_path, 'r') as f:
        reader = csv.reader(f)
        # Skip header if present
        header = next(reader, None)
        for row in reader:
            # Assume row format: timestamp, gx, gy, gz, ax, ay, az
            t = float(row[0])
            gx, gy, gz = map(float, row[1:4])
            ax, ay, az = map(float, row[4:7])
            imu_data.append((t, gx, gy, gz, ax, ay, az))
    return imu_data

def main():
    # Paths
    vocab_path = "data/orb_vocab.txt"
    orb_config_path = "data/orb_config_imu.yaml"  # config with IMU parameters
    obvi_config_path = "data/obvi_config.yaml"
    video_path = "data/sequence.mp4"
    imu_path = "data/imu_data.csv"

    # Initialize system with IMU sensor
    slam = ovs.OrbViSlam(
        orb_vocab_path=vocab_path,
        orb_config_path=orb_config_path,
        obvi_config_path=obvi_config_path,
        sensor=ovs.Sensor.IMU_MONOCULAR,  # monocular + IMU
        use_viewer=False
    )

    # Camera calibration (same as before)
    fx, fy, cx, cy = 718.856, 718.856, 607.1928, 185.2157
    intrinsics = {0: np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)}
    extrinsics = {0: np.eye(4, dtype=np.float64)}
    slam.set_camera_intrinsics(intrinsics)
    slam.set_camera_extrinsics(extrinsics)

    # Read IMU data
    imu_measurements = read_imu_data(imu_path)
    print(f"Loaded {len(imu_measurements)} IMU measurements")

    # Process video frames, synchronising with IMU
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    imu_idx = 0
    prev_timestamp = None
    imu_buffer = []

    print("Processing frames with IMU...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame timestamp (from video metadata or external source)
        # Here we assume a constant frame rate, but in practice you'd read from file.
        timestamp = frame_idx / 30.0  # replace with actual timestamp

        # Collect all IMU measurements that occurred between previous and current frame
        imu_buffer = []
        while imu_idx < len(imu_measurements) and imu_measurements[imu_idx][0] <= timestamp:
            if prev_timestamp is None or imu_measurements[imu_idx][0] > prev_timestamp:
                imu_buffer.append(imu_measurements[imu_idx])
            imu_idx += 1

        # Pass frame and IMU measurements to the system
        slam.process_frame(frame, timestamp, imu_buffer)

        prev_timestamp = timestamp
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames, used {len(imu_buffer)} IMU measurements")

    cap.release()
    print(f"Total frames processed: {frame_idx}")

    # Run final optimization
    print("Running final optimization...")
    slam.finalize()
    print("Optimization finished.")

    # Retrieve results
    traj = slam.get_trajectory()
    obj_map = slam.get_object_map()

    # Save trajectory (TUM format)
    with open("trajectory_imu.txt", "w") as f:
        for fid, pose in sorted(traj.items()):
            t = pose.translation
            q = pose.rotation_quaternion
            # Use the frame timestamp if available, else compute from frame index
            f.write(f"{fid*0.033:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    # Save object map
    with open("objects_imu.txt", "w") as f:
        for oid, (cls, state) in obj_map.items():
            c = state.pose.translation
            d = state.dimensions
            f.write(f"{oid} {cls} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                    f"{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}\n")

    print("Done. Results saved to trajectory_imu.txt and objects_imu.txt")

if __name__ == "__main__":
    main()