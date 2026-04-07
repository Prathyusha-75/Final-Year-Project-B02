# Quick Summary

## What this project does

Night Vision Seekers enhances low-light images and videos with a Zero-DCE neural network and a Gradio interface. It supports:

- Single-image enhancement
- Full-video enhancement
- Live webcam enhancement
- Recording enhanced webcam output to MP4

## How the project works

### 1. Model loading

The enhancement model lives in [models/zero_dce/model.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/models/zero_dce/model.py). It defines `enhance_net_nopool`, a lightweight convolutional network based on Zero-DCE.

- 7 convolution layers are used to predict curve parameters
- The network outputs 24 channels, split into 8 RGB curve maps
- The input image is refined over 8 enhancement iterations
- The model is configured with `number_f = 32` to match the shipped weights

Pretrained weights are loaded from `models/zero_dce/weights.pth`.

### 2. Device selection and inference

[processors/zero_dce.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/processors/zero_dce.py) wraps the model in `ZeroDCEProcessor`.

Its responsibilities are:

- Pick the best available device: `MPS -> CUDA -> CPU`
- Load weights and switch the model to evaluation mode
- Convert OpenCV BGR frames to normalized RGB tensors
- Run inference with `torch.no_grad()`
- Convert the output back to `uint8` BGR for OpenCV and Gradio

If the model fails to initialize, the processor falls back to returning the original frame.

### 3. Image enhancement pipeline

The main image workflow is implemented in [app.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/app.py).

When a user uploads an image, the app runs this pipeline:

1. Validate and normalize the input image format
2. Apply CLAHE on the LAB lightness channel to improve local contrast
3. Run Zero-DCE on the contrast-enhanced image
4. Blend classical enhancement and AI output
5. Apply optional user-controlled adjustments:
   - brightness boost
   - contrast boost
   - denoising
   - sharpness boost
6. Add a slight saturation boost
7. Return:
   - the enhanced image
   - a side-by-side comparison image
   - a status message

This means the project is not pure model inference only; it combines classical image processing with learned enhancement for more controllable results.

### 4. Video enhancement flow

The video tab processes uploaded files frame by frame inside `enhance_video_advanced()` in [app.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/app.py).

The flow is:

1. Open the uploaded file with OpenCV
2. Read FPS, width, height, and total frame count
3. Create an output writer in `outputs/night_enhanced_gradio.mp4`
4. Enhance frames through `processor.process_frame()`
5. Write the result using H264 when available, otherwise fall back to `mp4v` or `MJPG`
6. Return the saved output video and status text

The `fps_reduction` control lets the app process every Nth frame to reduce compute cost.

### 5. Live webcam enhancement

The live tab also lives in [app.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/app.py).

It uses Gradio webcam streaming plus a few module-level flags:

- `CAPTURE_ENABLED`
- `RECORDING_ENABLED`
- `RECORDED_FRAMES`
- `STATUS_OVERRIDE`

How it behaves:

- The webcam stream is always visible
- AI enhancement only runs while capture is enabled
- Pressing `Record` stores enhanced RGB frames in memory
- Pressing `Stop` saves buffered frames to `outputs/live_enhanced_<timestamp>.mp4`

This design keeps the live preview responsive while still allowing the enhanced output to be exported.

### 6. Training flow

[train.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/train.py) provides a simple retraining script.

Training works like this:

- Images are read from `data/train/`
- Each image is resized to `256x256`
- The model is trained for 30 epochs with Adam
- A lightweight exposure loss pushes output brightness toward `0.6`
- Weights are saved back to `models/zero_dce/weights.pth`

This is a minimal training loop intended for experimentation rather than a full research pipeline.

## Main files and responsibilities

- [app.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/app.py): Gradio UI and all app-side processing flows
- [processors/zero_dce.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/processors/zero_dce.py): device selection, model init, frame inference
- [models/zero_dce/model.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/models/zero_dce/model.py): Zero-DCE network definition
- [main.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/main.py): desktop webcam preview with side-by-side OpenCV window
- [train.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/train.py): retraining entry point
- [run_app.py](/e:/night_vision_with_gradio-main_gpu%20(2)/night_vision_with_gradio-main_gpu/run_app.py): helper launcher for the Gradio app

## In one sentence

This project loads a pretrained Zero-DCE model, wraps it in an OpenCV and Gradio pipeline, and combines neural enhancement with classical image-processing controls for images, videos, and live webcam input.
