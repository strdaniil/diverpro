import numpy as np
import cv2


v_path = "videos/ginz_dive.mp4"
t_path = "trajectories/ginz_dive_hips.npy"
o_path = "ginz_dive_output_overlay.mp4"
c = (0, 0, 255)  # red

def overlay_trajectory_on_video(video_path, trajectory_path, output_path, color=(0, 0, 255), radius=5):
    trajectory = np.load(trajectory_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(len(trajectory)):
        ret, frame = cap.read()
        if not ret or i >= len(trajectory):
            break

        x_norm, y_norm = trajectory[i]
        if not np.isnan(x_norm) and not np.isnan(y_norm):
            x = int(x_norm * width)
            y = int(y_norm * height)
            cv2.circle(frame, (x, y), radius, color, -1)


            for j in range(1, i):
                x1, y1 = trajectory[j - 1]
                x2, y2 = trajectory[j]
                if not np.any(np.isnan([x1, y1, x2, y2])):
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))
                    cv2.line(frame, pt1, pt2, color, thickness=2)

        out.write(frame)


    cap.release()
    out.release()
    print(f"Overlay saved to {output_path}")
