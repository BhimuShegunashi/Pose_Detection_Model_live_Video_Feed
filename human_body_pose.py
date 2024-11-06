import pickle
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initialize mediapipe pose class and drawing utilities
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculatAngle(point1, point2, point3):
    angle = math.degrees(
        math.atan2(point3[1] - point2[1], point3[0] - point2[0]) -
        math.atan2(point1[1] - point2[1], point1[0] - point2[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

# Function to detect pose landmarks
def detect_pose(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
        )
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))
    return output_image, landmarks

# Function to classify yoga poses
def classify_Pose(landmarks, output_image):
    label = 'Unknown Pose'
    color = (0, 0, 255)

    left_elbow_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_shoulder_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_knee_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_hip_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    right_hip_angle = calculatAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    
    
    
    # Checking the condition for Power pose
    
    # Check the both elbow angle is 45 to 90 degree
    if 45 < left_elbow_angle < 90 or 45 < right_elbow_angle < 90:
        if 80 < left_shoulder_angle < 110 or 80 < right_shoulder_angle < 110:
            label = 'Power Pose'
            print("Power Pose")
            
    # Checking the condition for T pose
    
    # Check the both elbow angle is  180 degree
    if 165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195:
        # Check the both shoulder angle is 90 degree
        if 75 < left_shoulder_angle < 110 and 75 < right_shoulder_angle < 110:
            label = 'T Pose'
            print("T Pose")
            
    # Checking the condition for Namaskaram pose
    
    # Check the both shoulder angle is 180 degree
    if 165 < left_shoulder_angle < 195 and 165 < right_shoulder_angle < 195:
        # Check both elbow angle is 180 degree
        if 145 < left_elbow_angle < 190 and 145 < right_elbow_angle < 190:
                label = 'Namaskaram pose'
                print("Namaskaram Pose")

    

    if label != 'Unknown Pose':
        color = (0, 255, 0)

    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return output_image, label

# For live video feed (webcam)
video = cv2.VideoCapture(0)
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
time_previous = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    frame, landmarks = detect_pose(frame, pose_video)
    if landmarks:
        frame, _ = classify_Pose(landmarks, frame)
    time_current = time()

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break
    
'''# Add this block to save the landmarks
if landmarks:
    # Save landmarks to a file after processing a frame
    with open('landmarks.pkl', 'wb') as file:
        pickle.dump(landmarks, file)'''

video.release()
cv2.destroyAllWindows()

