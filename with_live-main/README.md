# Night Vision Seekers

Low-light image, video, and webcam enhancement built with Zero-DCE, OpenCV, PyTorch, and Gradio.

## What you can do

- Enhance a single dark image in the browser
- Process an uploaded video frame by frame
- Run live webcam enhancement in the Gradio UI
- Record the enhanced live stream to MP4
- Retrain the shipped Zero-DCE model on your own low-light images

## Project docs

- [QUICK_SUMMARY.md](QUICK_SUMMARY.md): explains how the project works internally
- `README.md`: setup, usage, file layout, and troubleshooting

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- Pillow
- Gradio

Install the Python packages from `requirements.txt`, then install the correct PyTorch build for your machine.

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install PyTorch

For Windows with NVIDIA GPU, the repo includes:

```bash
powershell -ExecutionPolicy Bypass -File .\install_gpu_torch.ps1
```

For CPU-only installs, use:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

## Run the app

### Recommended: Gradio web app

Windows:

```bash
python run_app.py
```

Or run the app directly:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal. The app looks for an available port starting from `7860`.

### OpenCV webcam preview

```bash
python main.py
```

Press `q` to close the window.

## Gradio features

### Image Enhancement

- Upload an image
- Adjust brightness, contrast, sharpness, and denoise settings
- Get an enhanced result and a side-by-side comparison

### Video Enhancement

- Upload a video
- Optionally reduce effective FPS to lower processing cost
- Save the enhanced output to the `outputs/` folder

### Live Enhancement

- Start webcam capture inside Gradio
- View live and enhanced frames side by side
- Record the enhanced output
- Save the recording as MP4

## Train the model

1. Put training images in `data/train/`
2. Run:

```bash
python train.py
```

The script resizes images to `256x256`, trains for 30 epochs, and saves weights to `models/zero_dce/weights.pth`.

## Project structure

```text
night_vision_with_gradio-main_gpu/
|-- app.py
|-- main.py
|-- train.py
|-- run_app.py
|-- run_app.sh
|-- requirements.txt
|-- QUICK_SUMMARY.md
|-- install_gpu_torch.ps1
|-- models/
|   `-- zero_dce/
|       |-- model.py
|       `-- weights.pth
|-- processors/
|   |-- zero_dce.py
|   `-- enhancer.py
|-- data/
|   `-- train/
`-- outputs/
```

## Key implementation notes

- The app automatically selects `MPS`, then `CUDA`, then `CPU`
- Pretrained weights are loaded from `models/zero_dce/weights.pth`
- The main browser experience is defined in `app.py`
- The enhancement flow combines classical image processing with Zero-DCE inference

## Troubleshooting

### PyTorch is missing

Install a compatible PyTorch build first, then rerun:

```bash
pip install -r requirements.txt
```

### Gradio or OpenCV import errors

Make sure the virtual environment is activated and reinstall dependencies:

```bash
pip install -r requirements.txt
```

### GPU is not being used

The app automatically falls back to CPU if `MPS` or `CUDA` is unavailable.

### Webcam or output video does not save correctly

- Make sure the `outputs/` folder is writable
- Some systems may not support H264 through OpenCV, so the app falls back to `mp4v` or `MJPG`

## Reference

- Zero-DCE paper: https://arxiv.org/abs/2001.06826
- PyTorch docs: https://pytorch.org/docs
- Gradio docs: https://www.gradio.app/docs
