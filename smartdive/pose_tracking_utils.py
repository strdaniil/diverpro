import cv2
import mediapipe as mp
import numpy as np

JOINT_MAP = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "nose": 0,
    "shoulders": (11, 12),
    "hips": (23, 24),
    "knees": (25, 26),
    "ankles": (27, 28),
    "elbows": (13, 14),
    "wrists": (15, 16),
}

def get_joint_names():
    return list(JOINT_MAP.keys())

def track_joint(video_path, joint_name, min_conf=0.5, output_npy=None):
    mp_pose = mp.solutions.pose
    joint = JOINT_MAP.get(joint_name)

    if joint is None:
        raise ValueError(f"Unknown joint name: {joint_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    trajectory = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image)

            if not result.pose_landmarks:
                trajectory.append((np.nan, np.nan))
                continue

            lm = result.pose_landmarks.landmark

            if isinstance(joint, tuple):
              
                j1, j2 = joint
                if lm[j1].visibility > min_conf and lm[j2].visibility > min_conf:
                    x = (lm[j1].x + lm[j2].x) / 2
                    y = (lm[j1].y + lm[j2].y) / 2
                    trajectory.append((x, y))
                else:
                    trajectory.append((np.nan, np.nan))
            else:
           
                if lm[joint].visibility > min_conf:
                    trajectory.append((lm[joint].x, lm[joint].y))
                else:
                    trajectory.append((np.nan, np.nan))

    cap.release()
    trajectory = np.array(trajectory)

    if output_npy:
        np.save(output_npy, trajectory)

    return trajectory
