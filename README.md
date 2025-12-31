# 3D Point Cloud Change Detection (3DCDNet)
This repository provides a pipeline for detecting structural changes between 3D scenes using Intel RealSense depth camera video data. By capturing a reference scene (empty) and a current scene (with objects), the system generates a 3D reconstruction and highlights alterations in red.

# Hardware Requirements
Sensor: Intel RealSense D400 Series (D435, D435i, D455).

# Setup Instructions
1. Clone the Repository

git clone https://github.com/abinandha8086-lgtm/point-cloud-3d-change-detection.git
cd point-cloud-3d-change-detection

2. Create and Activate Environment
   
python3 -m venv open3d_env
source open3d_env/bin/activate  # On Windows: open3d_env\Scripts\activate

3. Install Requirements

pip install --upgrade pip
pip install -r requirements.txt

# Usage Guide

Step 1: Capture Scene Data

python3 capture_depth_pc.py
Controls: Press ENTER to start recording and 'Q' in the video window to stop and save.
        (Move the camera slowly in a slight arc to capture the sides of objects.)

Step 2: Run Change Detection
Process the recorded videos and handles the camera tracking, alignment, and change identification.

python3 run_camera_change.py


Step 3: Visualization
The script will automatically open an Open3D window once processing is complete:

Grey Points: Represent the original, unchanged structure.
Red Points: Represent the detected changes.


# Result example
<img width="1138" height="790" alt="Screenshot from 2025-12-31 12-05-36" src="https://github.com/user-attachments/assets/0144233c-6222-40ca-afdc-dabbce8c4af2" />


