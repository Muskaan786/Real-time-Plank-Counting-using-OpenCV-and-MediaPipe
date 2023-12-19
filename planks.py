import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize OpenCV VideoCapture for the camera (0 indicates the default camera)
cap = cv2.VideoCapture(0)

# Initialize variables
plank_duration = 0
plank_threshold = 10  # Adjust this threshold as needed
previous_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Pose
    results = pose.process(frame_rgb)

    # Check if the plank position is maintained
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark

        # Define the key landmark pairs for drawing lines
        line_pairs = [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                      (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                      (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
                      (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE)]

        # Calculate the angle between the shoulders, hips, and ankles
        correct_position = all(landmarks[a].y < landmarks[b].y for a, b in line_pairs)

        if correct_position:
            # Check for movement by comparing current landmarks with previous frame
            if previous_landmarks is not None:
                movement_detected = np.mean([np.linalg.norm([landmarks[i].x - previous_landmarks[i].x, landmarks[i].y - previous_landmarks[i].y]) for i in range(len(landmarks))])
                if movement_detected > 0.03:
                    correct_position = False
            previous_landmarks = [landmark for landmark in landmarks]

            # If in the correct position, draw a green outline and increment duration
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
            cv2.putText(frame, "Right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            plank_duration += 1

        else:
            # Draw a red point and display "Wrong" if not in the correct position
            cv2.circle(frame, (50, 50), 10, (0, 0, 255), -1)
            cv2.putText(frame, "Wrong", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            plank_duration = 0

        # Draw lines between key landmarks
        for a, b in line_pairs:
            pt1 = (int(landmarks[a].x * frame.shape[1]), int(landmarks[a].y * frame.shape[0]))
            pt2 = (int(landmarks[b].x * frame.shape[1]), int(landmarks[b].y * frame.shape[0]))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Plank Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
