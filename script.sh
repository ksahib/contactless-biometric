cd /home/kazisahib/biometric

~/.pyenv/versions/3.11.9/bin/python -m venv .venv311
source .venv311/bin/activate

pip install --upgrade pip
pip install opencv-python "mediapipe==0.10.14" "protobuf>=4.25.3,<5" fingerprint-enhancer pyfing keras

python main.py
