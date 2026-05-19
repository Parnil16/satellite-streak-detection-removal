#!/bin/bash
set -e

echo "========================================="
echo "1. Creating Python Virtual Environment..."
echo "========================================="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip unzip tmux
python3 -m venv ~/streak_env
source ~/streak_env/bin/activate

echo "========================================="
echo "2. Installing CUDA-enabled PyTorch..."
echo "========================================="
pip install --upgrade pip
# Install PyTorch with CUDA 12.4 support (works well with A30 and Ubuntu 24.04)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "========================================="
echo "3. Verifying GPU Access..."
echo "========================================="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Detected:', torch.cuda.get_device_name(0))
else:
    print('WARNING: GPU NOT DETECTED!')
"

echo "========================================="
echo "4. Installing ML Libraries..."
echo "========================================="
pip install segmentation-models-pytorch albumentations opencv-python pillow tqdm

echo "========================================="
echo "5. Extracting Dataset..."
echo "========================================="
if [ -f "dataset.zip" ]; then
    echo "Extracting dataset.zip..."
    unzip -q dataset.zip -d dataset/
    echo "Dataset extracted successfully!"
else
    echo "WARNING: dataset.zip not found in the current directory. Please ensure it is uploaded."
fi

echo "========================================="
echo "Setup Complete! Virtual Environment activated."
echo "To start training in tmux, run the following commands:"
echo "  tmux new -s train_session"
echo "  source ~/streak_env/bin/activate"
echo "  python train_ssh.py"
echo "========================================="
