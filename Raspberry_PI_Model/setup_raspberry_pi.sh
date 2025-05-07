#!/bin/bash




sudo apt-get update


sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7 \
    libtiff5


pip3 install --upgrade pip
pip3 install numpy
pip3 install tflite-runtime
pip3 install opencv-python

# Make Python scripts executable
chmod +x run_posture_model.py
chmod +x run_headless.py
chmod +x find_camera.py

echo "Installation complete! You can now run the posture detection model with:"
echo "- First, find your camera: python3 find_camera.py"
echo "- With display: python3 run_posture_model.py --camera X (replace X with your camera number)"
echo "- Headless mode: python3 run_headless.py --camera X (replace X with your camera number)" 