$ErrorActionPreference = "Stop"

Write-Host "Installing CUDA-enabled PyTorch for Windows/NVIDIA..."
Write-Host "Using the official PyTorch wheel index for CUDA 12.4."

python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "Verifying CUDA availability..."
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
