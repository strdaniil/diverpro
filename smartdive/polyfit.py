import numpy as np
import cv2

v_path = "videos/adive.mp4"
t_path = "trajectories/adive_hips.npy"
d = 3
o_path = "adive_polyfit_overlay.mp4"
c= (0, 255, 255)  

def fit_polynomial_time_param(trajectory, degree):
    """Fit x(t) and y(t) separately using time as parameter."""
    valid = ~np.isnan(trajectory).any(axis=1)
    trajectory = trajectory[valid]
    n = len(trajectory)
    
    t = np.arange(n)
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    if n < degree + 1:
        raise ValueError(f"Not enough valid points to fit degree {degree} polynomial.")

    coeffs_x = np.polyfit(t, x, degree)
    coeffs_y = np.polyfit(t, y, degree)
    return coeffs_x, coeffs_y

def overlay_fit_curve(video_path, trajectory_path, degree, output_path, color):

    trajectory = np.load(trajectory_path)
    coeffs_x, coeffs_y = fit_polynomial_time_param(trajectory, degree)
    poly_x = np.poly1d(coeffs_x)
    poly_y = np.poly1d(coeffs_y)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        for j in range(1, i):
            x1, y1 = poly_x(j - 1), poly_y(j - 1)
            x2, y2 = poly_x(j), poly_y(j)

            if not np.any(np.isnan([x1, y1, x2, y2])):
                pt1 = (int(x1 * width), int(y1 * height))
                pt2 = (int(x2 * width), int(y2 * height))
                cv2.line(frame, pt1, pt2, color, thickness=2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[✓] Saved to {output_path}")

