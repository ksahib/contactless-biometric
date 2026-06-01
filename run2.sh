python -m pip uninstall -y tensorflow tensorflow-cpu keras nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]==2.21.0"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
