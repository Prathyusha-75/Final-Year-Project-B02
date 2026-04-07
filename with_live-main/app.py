#!/usr/bin/env python3
"""
Night Vision Seekers - Gradio Web Interface
Low-light image enhancement using Zero-DCE with enhanced features
"""

import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
from processors.zero_dce import ZeroDCEProcessor
from processors.enhancer import ImageEnhancer
import threading
import time

BASE_DIR = Path(__file__).resolve().parent

# Initialize processor
processor = ZeroDCEProcessor(weights_path=BASE_DIR / "models" / "zero_dce" / "weights.pth")
init_success = processor.initialize()

if not init_success:
    print("⚠️  Warning: Starting with untrained model")

# Global variables for video processing
processing_progress = {"count": 0, "total": 0}
progress_lock = threading.Lock()

# Live enhancement control flag.
# Using a module-level flag is more reliable than gating with gr.State for stream callbacks.
CAPTURE_ENABLED = False
CAPTURE_LOCK = threading.Lock()

# Recording control + buffer (module-level for reliability with streaming callbacks)
RECORDING_ENABLED = False
RECORDING_LOCK = threading.Lock()
RECORDED_FRAMES = []

# One-time status override so stop/save messages aren't immediately overwritten
STATUS_OVERRIDE = None
STATUS_LOCK = threading.Lock()

def _recording_indicator_html(is_recording=False):
    state_class = "recording-indicator active" if is_recording else "recording-indicator"
    return f"""
    <div class="{state_class}">
        <span class="recording-dot"></span>
        <span class="recording-text">REC</span>
    </div>
    """

def _start_live_capture():
    global CAPTURE_ENABLED, RECORDING_ENABLED, RECORDED_FRAMES, STATUS_OVERRIDE
    with CAPTURE_LOCK:
        CAPTURE_ENABLED = True
    with RECORDING_LOCK:
        RECORDING_ENABLED = False
        RECORDED_FRAMES = []
    with STATUS_LOCK:
        STATUS_OVERRIDE = None
    print("[Live] Capture enabled")
    return "▶️ Capturing live video (enhancement running)...", _recording_indicator_html(False)

def _stop_live_capture():
    global CAPTURE_ENABLED, RECORDING_ENABLED, STATUS_OVERRIDE
    with CAPTURE_LOCK:
        CAPTURE_ENABLED = False
    with RECORDING_LOCK:
        RECORDING_ENABLED = False
    with STATUS_LOCK:
        STATUS_OVERRIDE = "⏸️ Capturing stopped."
    print("[Live] Capture disabled")
    return "⏸️ Capturing stopped.", _recording_indicator_html(False)

def _start_recording():
    global CAPTURE_ENABLED, RECORDING_ENABLED, RECORDED_FRAMES
    with CAPTURE_LOCK:
        CAPTURE_ENABLED = True
    with RECORDING_LOCK:
        RECORDING_ENABLED = True
        RECORDED_FRAMES = []
    print("[Live] Recording enabled")
    return "⏺️ Recording enhanced video...", _recording_indicator_html(True)

def _stop_recording_and_save():
    """
    Stop enhancement/recording, then save the buffered frames (if any).
    Returns: (video_path_or_None, status_text)
    """
    global CAPTURE_ENABLED, RECORDING_ENABLED, RECORDED_FRAMES, STATUS_OVERRIDE

    with CAPTURE_LOCK:
        CAPTURE_ENABLED = False
    with RECORDING_LOCK:
        RECORDING_ENABLED = False
        frames_to_save = list(RECORDED_FRAMES)
        RECORDED_FRAMES = []

    print(f"[Live] Saving recording ({len(frames_to_save)} frames)...")
    if len(frames_to_save) == 0:
        msg = "⏹️ Stopped (no recording to save)."
        with STATUS_LOCK:
            STATUS_OVERRIDE = msg
        return None, msg, _recording_indicator_html(False)

    # Reuse existing saver
    output_path, msg = save_recorded_video(frames_to_save)
    if output_path is None:
        with STATUS_LOCK:
            STATUS_OVERRIDE = msg
        return None, msg, _recording_indicator_html(False)
    with STATUS_LOCK:
        STATUS_OVERRIDE = msg
    return output_path, msg, _recording_indicator_html(False)

def _handle_live_stop():
    with RECORDING_LOCK:
        is_recording = RECORDING_ENABLED
    if is_recording:
        return _stop_recording_and_save()
    status_text, indicator_html = _stop_live_capture()
    return None, status_text, indicator_html

def enhance_image_with_options(input_image, brightness_boost=1.0, contrast_boost=1.0, 
                               sharpness_boost=1.0, denoise_strength=0):
    """
    IMPROVED image processing with better enhancement pipeline
    """
    if input_image is None:
        return None, None, "❌ No image provided"
    
    try:
        # Handle both PIL Image and numpy array
        if isinstance(input_image, np.ndarray):
            frame = input_image.copy()
        else:
            frame = np.array(input_image).copy()
        
        # Validate input
        if frame.size == 0:
            return None, None, "❌ Invalid image"
        
        # Ensure frame is in BGR format
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3 and frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        original_frame = frame.copy()
        
        # Step 1: CLAHE contrast enhancement (better than histogram eq)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 2: Apply AI enhancement (Zero-DCE)
        try:
            ai_enhanced = processor.process_frame(enhanced)
            if ai_enhanced is not None and ai_enhanced.size > 0:
                ai_enhanced = np.clip(ai_enhanced, 0, 255).astype(np.uint8)
                # Blend AI enhancement (40% weight) with CLAHE (60% weight)
                enhanced = cv2.addWeighted(enhanced, 0.6, ai_enhanced, 0.4, 0)
            else:
                enhanced = enhanced
        except:
            pass
        
        # Step 3: Apply user-controlled brightness boost (more conservative)
        if brightness_boost != 1.0:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,2] = hsv[:,:,2] * min(brightness_boost, 1.5)  # Cap at 1.5x
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Step 4: Apply user-controlled contrast boost (gamma correction)
        if contrast_boost != 1.0:
            gamma = 1.0 / max(contrast_boost, 0.5)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
        
        # Step 5: Apply denoising if requested
        if denoise_strength > 0:
            strength = int(max(3, min(denoise_strength, 21)))
            if strength % 2 == 0:
                strength += 1
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h=strength, 
                                                       templateWindowSize=7, searchWindowSize=21)
        
        # Step 6: Apply sharpening if requested (UnSharp Mask for better results)
        if sharpness_boost > 1.0:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.0 + (sharpness_boost - 1.0) * 0.3, gaussian, 
                                      -(sharpness_boost - 1.0) * 0.3, 0)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Step 7: Final color balance and saturation adjustment
        enhanced_hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        enhanced_hsv[:,:,1] = enhanced_hsv[:,:,1] * 1.1  # Slight saturation boost
        enhanced_hsv = np.clip(enhanced_hsv, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # Create comparison image
        h, w = frame.shape[:2]
        comparison = np.hstack((original_frame, enhanced))
        
        # Convert to RGB for Gradio
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        
        return enhanced_rgb, comparison_rgb, "✅ Enhancement successful"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

def enhance_live_frame(frame, brightness_boost=1.0, contrast_boost=1.0,
                       sharpness_boost=1.0, denoise_strength=0):
    """
    Process a single frame for live webcam enhancement.
    Enhancement runs continuously only while CAPTURE_ENABLED is True.
    """
    global STATUS_OVERRIDE

    if frame is None:
        return None, None, "❌ No frame provided"

    # Always show the live feed; only run AI when capturing.
    with CAPTURE_LOCK:
        enabled = CAPTURE_ENABLED

    if not enabled:
        # Return the raw frame on the right too, so it never looks "frozen".
        with STATUS_LOCK:
            override = STATUS_OVERRIDE
            # Consume the override once so future frames show the default status again.
            STATUS_OVERRIDE = None
        if override:
            return frame, frame, override
        return frame, frame, "⏸️ Capturing stopped (press Capture Live)"

    try:
        # Print occasionally to confirm the callback is running.
        # (Avoid spamming the terminal; use modulo of time.)
        if int(time.time() * 2) % 10 == 0:
            print("[Live] Enhancing frame...")

        # Gradio webcam frames typically come in RGB; convert to BGR
        bgr_frame = frame
        if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        enhanced_rgb, _, _status = enhance_image_with_options(
            bgr_frame,
            brightness_boost,
            contrast_boost,
            sharpness_boost,
            denoise_strength,
        )

        if enhanced_rgb is None:
            return frame, frame, "❌ Enhancement failed; showing live feed"

        # If recording, buffer the enhanced frames for later saving.
        with RECORDING_LOCK:
            if RECORDING_ENABLED:
                RECORDED_FRAMES.append(enhanced_rgb)

        return frame, enhanced_rgb, "✅ Enhancing live video"
    except Exception as e:
        print(f"Live frame processing error: {e}")
        return frame, frame, f"❌ Error enhancing live frame: {e}"



def save_recorded_video(recorded_frames):
    """
    Save recorded frames as a video file
    """
    if not recorded_frames or len(recorded_frames) == 0:
        return None, "❌ No frames recorded"
    
    try:
        output_path = f"outputs/live_enhanced_{int(time.time())}.mp4"
        Path("outputs").mkdir(exist_ok=True)
        
        # Get frame dimensions from first frame
        height, width = recorded_frames[0].shape[:2]
        
        # Use H264 codec for best Gradio compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))  # 10 FPS
        
        if not out.isOpened():
            # Try MP4V as fallback
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
        
        if not out.isOpened():
            return None, "❌ Failed to initialize video writer"
        
        for frame in recorded_frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()
        
        # Verify output file
        if not Path(output_path).exists():
            return None, "❌ Output file was not created"
        
        file_size = Path(output_path).stat().st_size
        if file_size == 0:
            return None, "❌ Output file is empty"
        
        return output_path, f"✅ Recorded video saved: {len(recorded_frames)} frames | File: {file_size/1024:.1f} KB"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error saving video: {str(e)}"


def enhance_video_advanced(video_file, fps_reduction=1):
    """
    Advanced video processing with H.264 codec for Gradio compatibility
    """
    if video_file is None:
        return None, "❌ No video provided"
    
    try:
        output_path = "outputs/night_enhanced_gradio.mp4"
        Path("outputs").mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            return None, "❌ Failed to open video file"
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = max(1, int(original_fps // fps_reduction))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if width == 0 or height == 0 or total_frames == 0:
            return None, "❌ Invalid video properties"
        
        # Use H264 codec for best Gradio compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Try MP4V as fallback
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Try MJPEG as last resort
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return None, "❌ Failed to initialize video writer"
        
        frame_count = 0
        written_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Validate frame
            if frame is None or frame.size == 0:
                continue
            
            frame_count += 1
            
            # Process only every Nth frame based on fps_reduction
            if frame_count % fps_reduction == 0:
                try:
                    enhanced = processor.process_frame(frame)
                    
                    # Validate enhanced frame
                    if enhanced is None or enhanced.size == 0:
                        enhanced = frame
                    
                    # Ensure proper data type
                    if enhanced.dtype != np.uint8:
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                    
                    # Ensure correct shape
                    if enhanced.shape[:2] != (height, width):
                        enhanced = cv2.resize(enhanced, (width, height))
                    
                    # Ensure BGR format
                    if len(enhanced.shape) == 2:
                        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    
                    out.write(enhanced)
                    written_count += 1
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    out.write(frame)
                    written_count += 1
        
        cap.release()
        out.release()
        
        if written_count == 0:
            return None, "❌ No frames were processed"
        
        # Verify output file exists and has size
        if not Path(output_path).exists():
            return None, "❌ Output file was not created"
        
        file_size = Path(output_path).stat().st_size
        if file_size == 0:
            return None, "❌ Output file is empty"
        
        return output_path, f"✅ Video processed: {written_count} frames at {fps} FPS | File: {file_size/1024:.1f} KB"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"

def ensure_gradio_video_compatibility(video_path):
    """
    Ensure video file is compatible with Gradio playback
    """
    try:
        video_path = str(video_path)
        if not Path(video_path).exists():
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return video_path
    except:
        return None

def get_device_info():
    """
    Get information about the processing device
    """
    if processor.model is None:
        return "Processing Device: Not initialized"
    return f"Processing Device: {processor.get_device_name()}"

# CSS for better styling
css = """
.container { max-width: 1200px; margin: 0 auto; }
.comparison-container { display: flex; gap: 10px; }
.recording-indicator {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    min-height: 40px;
    padding: 10px 14px;
    border-radius: 999px;
    border: 1px solid rgba(185, 28, 28, 0.18);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(255, 244, 244, 0.9));
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    color: #991b1b;
    font-weight: 700;
    letter-spacing: 0.18em;
    font-size: 0.78rem;
    opacity: 0;
    transform: translateY(-4px);
    transition: opacity 0.2s ease, transform 0.2s ease;
    pointer-events: none;
}
.recording-indicator.active {
    opacity: 1;
    transform: translateY(0);
}
.recording-dot {
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: radial-gradient(circle at 35% 35%, #fca5a5 0%, #ef4444 45%, #b91c1c 100%);
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6);
    animation: recording-pulse 1.4s ease-out infinite;
}
.recording-text {
    font-family: Arial, sans-serif;
}
@keyframes recording-pulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.45); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}
"""

# Create Gradio Interface
with gr.Blocks(title="Night Vision Seekers") as interface:
    gr.Markdown("""
    # 🌙 Night Vision Seekers
    ### Low-Light Image Enhancement using Zero-DCE Neural Network
    
    Transform dark and underexposed images into clear, bright visuals using advanced deep learning.
    """)
    
    gr.Markdown(f"**{get_device_info()}**")
    
    with gr.Tabs():
        # Image Enhancement Tab
        with gr.Tab("📷 Image Enhancement"):
            gr.Markdown("### Upload a dark image to enhance it with AI")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Input Image",
                        type="numpy"
                    )
                    
                    gr.Markdown("### Enhancement Settings")
                    brightness_slider = gr.Slider(
                        label="Brightness Boost",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    contrast_slider = gr.Slider(
                        label="Contrast Boost",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    sharpness_slider = gr.Slider(
                        label="Sharpness Boost",
                        minimum=1.0,
                        maximum=3.0,
                        value=1.0,
                        step=0.1
                    )
                    denoise_slider = gr.Slider(
                        label="Denoise Strength",
                        minimum=0,
                        maximum=20,
                        value=0,
                        step=1
                    )
                    
                    enhance_btn = gr.Button(
                        "✨ Enhance Image",
                        scale=1,
                        variant="primary"
                    )
                
                with gr.Column():
                    output_image = gr.Image(
                        label="Enhanced Image",
                        type="numpy"
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    comparison_image = gr.Image(
                        label="Side-by-Side Comparison (Original | Enhanced)",
                        type="numpy"
                    )
            
            enhance_btn.click(
                enhance_image_with_options,
                inputs=[input_image, brightness_slider, contrast_slider, 
                       sharpness_slider, denoise_slider],
                outputs=[output_image, comparison_image, status_text]
            )
        
        # Video Enhancement Tab
        with gr.Tab("🎬 Video Enhancement"):
            gr.Markdown("### Upload a video to enhance all frames")
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input Video")
                    fps_reduction = gr.Slider(
                        label="FPS Reduction (1=normal, 2=half FPS, etc)",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1
                    )
                    video_btn = gr.Button(
                        "✨ Process Video",
                        variant="primary"
                    )
                
                with gr.Column():
                    video_output = gr.Video(
                        label="Enhanced Video",
                        format="mp4",
                        interactive=False
                    )
                    video_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        lines=3
                    )
            
            video_btn.click(
                fn=enhance_video_advanced,
                inputs=[video_input, fps_reduction],
                outputs=[video_output, video_status],
                show_progress=True
            )
        
        # Live Enhancement Tab
        with gr.Tab("📹 Live Enhancement"):
            gr.Markdown("### Real-time webcam enhancement with AI")
            
            with gr.Row():
                with gr.Column():
                    # streaming=True is required for continuous webcam frame updates
                    webcam_input = gr.Image(
                        label="Live Camera Feed",
                        sources=["webcam"],
                        streaming=True,
                    )
                    
                    gr.Markdown("### Enhancement Settings")
                    live_brightness_slider = gr.Slider(
                        label="Brightness Boost",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    live_contrast_slider = gr.Slider(
                        label="Contrast Boost", 
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    live_sharpness_slider = gr.Slider(
                        label="Sharpness Boost",
                        minimum=1.0,
                        maximum=3.0,
                        value=1.0,
                        step=0.1
                    )
                    live_denoise_slider = gr.Slider(
                        label="Denoise Strength",
                        minimum=0,
                        maximum=20,
                        value=0,
                        step=1
                    )
                    
                    with gr.Row():
                        capture_btn = gr.Button("📸 Capture Live", variant="primary")
                        record_btn = gr.Button("⏺️ Record", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                
                with gr.Column():
                    recording_indicator = gr.HTML(
                        value=_recording_indicator_html(False),
                        show_label=False
                    )
                    with gr.Row():
                        live_original = gr.Image(
                            label="Live Feed",
                            type="numpy",
                        )
                        live_output = gr.Image(
                            label="Enhanced (AI)",
                            type="numpy",
                        )
                    live_status = gr.Textbox(
                        label="Status",
                        value="Press Capture Live to enhance live. Press Record to save MP4. Press Stop to save.",
                        interactive=False
                    )
                    recorded_video_output = gr.Video(
                        label="Recorded Enhanced Video",
                        format="mp4",
                        interactive=False,
                        buttons=["download"],
                    )

            capture_btn.click(
                fn=lambda: _start_live_capture(),
                inputs=[],
                outputs=[live_status, recording_indicator],
            )
            record_btn.click(
                fn=lambda: _start_recording(),
                inputs=[],
                outputs=[live_status, recording_indicator],
            )
            stop_btn.click(
                fn=_handle_live_stop,
                inputs=[],
                outputs=[recorded_video_output, live_status, recording_indicator],
            )

            # Continuous live enhancement on incoming webcam frames (model only runs while capturing)
            webcam_input.stream(
                fn=enhance_live_frame,
                inputs=[
                    webcam_input,
                    live_brightness_slider,
                    live_contrast_slider,
                    live_sharpness_slider,
                    live_denoise_slider,
                ],
                outputs=[live_original, live_output, live_status],
                # Give the model time to finish; avoid cancelling in-flight work.
                stream_every=0.8,
                trigger_mode="multiple",
                concurrency_limit=1,
            )
        
        # Advanced Settings Tab
        with gr.Tab("⚙️ Advanced Settings"):
            gr.Markdown("""
            ### About the Enhancement
            
            **Technology:** Zero-DCE (Deep Curve Estimation)
            - Self-supervised learning approach
            - No paired training data required
            - Optimized for low-light enhancement
            
            **Enhancements Applied:**
            1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            2. AI-based Zero-DCE enhancement
            3. User-controlled brightness adjustment
            4. Gamma correction for contrast
            5. Optional denoising (NLM - Non-Local Means)
            6. Unsharp masking for sharpness
            7. Color saturation adjustment
            
            **Tips:**
            - Start with default settings for best results
            - Adjust brightness for very dark images
            - Use denoise for grainy/noisy images
            - Increase sharpness for detail enhancement
            """)

if __name__ == "__main__":
    import socket
    
    # Find available port
    def find_available_port(start_port=7860, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                continue
        return start_port
    
    port = find_available_port()
    print(f"\n🚀 Starting Gradio on port {port}...")
    print(f"📍 Open browser: http://localhost:{port}")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=css
    )
