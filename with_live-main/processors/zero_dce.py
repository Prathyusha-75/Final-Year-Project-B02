import cv2
import torch
import numpy as np
from pathlib import Path
from models.zero_dce.model import enhance_net_nopool

class ZeroDCEProcessor:
    def __init__(self, weights_path: str = None):
        self.device = self._select_device()
            
        self.weights_path = Path(weights_path).expanduser().resolve() if weights_path else None
        self.model = None
    
    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def get_device_name(self) -> str:
        if self.device.type == "cuda":
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        if self.device.type == "mps":
            return "Apple Metal (MPS)"
        return "CPU"

    def initialize(self) -> bool:
        try:
            self.model = enhance_net_nopool().to(self.device)
            if self.weights_path:
                if not self.weights_path.exists():
                    raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
                state = torch.load(str(self.weights_path), map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                print(f"Weights loaded onto {self.device}")
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
            print(f"Using processing device: {self.get_device_name()}")
            self.model.eval()
            return True
        except Exception as e:
            print(f"Init Error: {e}")
            self.model = None
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None: return frame
        
        # BGR to RGB and Normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            enhanced = self.model(tensor)

        # Back to CPU and Numpy
        enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_uint8 = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
