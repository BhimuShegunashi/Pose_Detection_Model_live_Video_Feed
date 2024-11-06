# Pose_Detection_Model_live_Video_Feed
# Overview <div>
This repository contains a Python-based implementation of a pose detection model using OpenCV and MediaPipe. The project aims to detect human body poses and classify them into predefined yoga or fitness poses such as the Power Pose, T Pose, and Namaskaram Pose by analyzing landmark angles of key body joints. The code supports real-time pose detection using a webcam.
<div>
# Features <div>
Real-time Pose Detection: Uses a webcam to detect poses live.
Pose Classification: Classifies detected poses into Power Pose, T Pose, or Namaskaram Pose based on calculated angles of specific body joints.
Angle Calculation: Computes angles between detected keypoints to determine pose type.
Mediapipe Integration: Utilizes Google's MediaPipe for pose detection and OpenCV for image processing.
</div>
Requirements<div>
Ensure you have Python 3.x installed and set up the following libraries:<div>

mediapipe<div>
opencv-python<div>
numpy<div>
pickle<div>
Interaction:<div>

The live video feed will be displayed in a resizable window.<div>
Detected poses will be shown on the screen.<div>
Press the Esc key to exit the video feed.<div>
