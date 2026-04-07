import cv2
import numpy as np
from pathlib import Path
from processors.zero_dce import ZeroDCEProcessor

def main():
    base_dir = Path(__file__).resolve().parent
    processor = ZeroDCEProcessor(weights_path=base_dir / "models" / "zero_dce" / "weights.pth")
    if not processor.initialize():
        print("Starting with untrained model...")
    else:
        print(f"Night Vision Seeker running on {processor.get_device_name()}")

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Enhancement
        enhanced = processor.process_frame(frame)
        
        # Side-by-side display
        display = np.hstack((frame, enhanced))
        cv2.imshow('Night Vision Seeker (Left: Original | Right: AI Enhanced)', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
