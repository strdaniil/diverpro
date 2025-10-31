import cv2
import mediapipe as mp
import math

def calculate_angle(a, b, c):

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = dot_product / (mag_ba * mag_bc)
    angle = math.acos(max(min(cos_angle, 1.0), -1.0))  
    return math.degrees(angle)

def analyze_knee(image_path, facing="left", show=True, save_path=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        print("No pose detected.")
        return

    landmarks = result.pose_landmarks.landmark
    h, w, _ = image.shape


    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]


    if facing == "left":
        is_left_front = left_ankle.x < right_ankle.x
    elif facing == "right":
        is_left_front = left_ankle.x > right_ankle.x
    else:
        print("Invalid 'facing' parameter. Use 'left' or 'right'.")
        return

    if is_left_front:
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = left_ankle
        side = "Left"
    else:
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = right_ankle
        side = "Right"


    hip_pt = (int(hip.x * w), int(hip.y * h))
    knee_pt = (int(knee.x * w), int(knee.y * h))
    ankle_pt = (int(ankle.x * w), int(ankle.y * h))


    angle = calculate_angle(hip_pt, knee_pt, ankle_pt)
    if angle is None:
        print("Cannot compute angle.")
        return

    print(f"{side} knee angle: {angle:.1f}°")

    # Feedback
    if angle < 125:
        print("Too crouched. Straighten your front leg to raise your hips.")
    elif angle > 150:
        print("Too straight. Slightly bend your front leg for better loading.")
    else:
        print("Knee angle is within ideal range.")

    cv2.circle(image, hip_pt, 6, (255, 0, 0), -1)      
    cv2.circle(image, knee_pt, 6, (0, 0, 255), -1)     
    cv2.circle(image, ankle_pt, 6, (0, 255, 255), -1)  


    cv2.line(image, hip_pt, knee_pt, (0, 255, 0), 2)
    cv2.line(image, ankle_pt, knee_pt, (0, 255, 0), 2)


    cv2.putText(image, f"{angle:.1f} deg", (knee_pt[0]+10, knee_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if show:
        cv2.imshow("Knee Angle", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    analyze_knee("images/test2.jpg", facing="right", show=True, save_path="knee_output.png")
