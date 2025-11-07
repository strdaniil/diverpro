from pose_tracking_utils import track_joint
from overlay import overlay_trajectory_on_video
from polyfit import overlay_fit_curve

traj = track_joint("videos/adive.mp4", "ankles", output_npy="trajectories/adive_hips.npy")
video_path = "videos/adive.mp4"
trajectory_path = "trajectories/adive_hips.npy"
output_path = "adive_output_overlay.mp4"
color = (0, 0, 255)
overlay_trajectory_on_video(video_path, trajectory_path, output_path, color)



video_path = "videos/adive.mp4"
trajectory_path = "trajectories/adive_hips.npy"
degree = 3
output_path = "polyfit_overlay.mp4"
color = (0, 255, 255)
overlay_fit_curve(video_path, trajectory_path, degree, output_path, color)
