
# Hand Landmark Detection System

This project implements real-time detection of hand landmarks (such as fingertips and finger joints) using Mediapipe and OpenCV.

## Overview
This system detects and visualizes 21 keypoints on each hand, allowing further development of gesture recognition, virtual interaction, and other interactive computer vision applications.

## Features
- Real-time hand landmark detection using webcam
- Visualization of detected hand landmarks
- Identification and marking of fingertips

## Requirements
- Python 3.6 or higher
- OpenCV
- Mediapipe
- NumPy

## Installation

1. Clone the repository:
git clone https://github.com/deepikaksr/hand-landmark-detection.git cd hand-landmark-detection

2. Set the dependencies:
pip install -r requirements.txt

## Usage
Run the hand landmark detection script using : python3 hand_landmark_detection.py

Press `q` to quit the webcam window.


## Hand Landmark Index Reference (Mediapipe): 

| Landmark ID | Landmark Description |
|-------------|----------------------|
| 0           | Wrist                |
| 1           | Thumb_CMC            |
| 2           | Thumb_MCP            |
| 3           | Thumb_IP             |
| 4           | Thumb_Tip            |
| 5           | Index_Finger_MCP     |
| 6           | Index_Finger_PIP     |
| 7           | Index_Finger_DIP     |
| 8           | Index_Finger_Tip     |
| 9           | Middle_Finger_MCP    |
| 10          | Middle_Finger_PIP    |
| 11          | Middle_Finger_DIP    |
| 12          | Middle_Finger_Tip    |
| 13          | Ring_Finger_MCP      |
| 14          | Ring_Finger_PIP      |
| 15          | Ring_Finger_DIP      |
| 16          | Ring_Finger_Tip      |
| 17          | Pinky_MCP            |
| 18          | Pinky_PIP            |
| 19          | Pinky_DIP            |
| 20          | Pinky_Tip            |
