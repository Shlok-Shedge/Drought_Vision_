import torch
import numpy as np
from src.model import MultispectralUNet
from src.utils import calculate_ndvi, compute_drought_statistics, mask_to_rgb, mask_to_rgba

class DroughtInferenceEngine:
    def __init__(self, model_weights_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = MultispectralUNet(in_channels=4, num_classes=6)
        self.has_weights = False
        self.load_error = None

        if model_weights_path:
            import os
            # Resolve relative path against project root if needed
            target_path = model_weights_path
            if not os.path.isabs(target_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                candidate = os.path.join(project_root, target_path)
                if os.path.exists(candidate):
                    target_path = candidate

            try:
                state_dict = torch.load(target_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.has_weights = True
                print(f"[DroughtVision AI] Loaded trained weights from: {target_path}")
            except Exception as e:
                self.load_error = str(e)
                print(f"[DroughtVision AI] Notice: Could not load weights ({e}). Running hybrid DSI inference mode.")
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, bands_chw, region_name=""):
        """
        Runs inference on multispectral bands.
        bands_chw: numpy array of shape [4, H, W] (Blue, Green, Red, NIR) normalized to [0, 1]
        Returns:
            mask: [H, W] integer class mask (0 to 5)
            mask_rgb: [H, W, 3] uint8 colorized mask
            stats: list of drought category percentages
            ndvi: [H, W] float NDVI map
            mask_rgba: [H, W, 4] transparently clipped district RGBA mask
        """
        blue, green, red, nir = bands_chw[0], bands_chw[1], bands_chw[2], bands_chw[3]
        ndvi = calculate_ndvi(nir, red)

        if self.has_weights:
            # Model-based deep learning inference
            tensor_in = torch.from_numpy(bands_chw).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
            mask = preds
        else:
            # Baseline Remote Sensing DSI rule mapping (ensures live demo works even before training on Colab)
            # DSI mapped from NDVI and NIR dryness
            mask = np.zeros_like(ndvi, dtype=np.int32)
            # 0: No Drought (NDVI > 0.50)
            # 1: D0 Abnormally Dry (0.38 < NDVI <= 0.50)
            # 2: D1 Moderate (0.28 < NDVI <= 0.38)
            # 3: D2 Severe (0.18 < NDVI <= 0.28)
            # 4: D3 Extreme (0.10 < NDVI <= 0.18)
            # 5: D4 Exceptional (NDVI <= 0.10)
            mask[ndvi > 0.50] = 0
            mask[(ndvi > 0.38) & (ndvi <= 0.50)] = 1
            mask[(ndvi > 0.28) & (ndvi <= 0.38)] = 2
            mask[(ndvi > 0.18) & (ndvi <= 0.28)] = 3
            mask[(ndvi > 0.10) & (ndvi <= 0.18)] = 4
            mask[ndvi <= 0.10] = 5

        mask_rgb = mask_to_rgb(mask)
        mask_rgba = mask_to_rgba(mask, region_name=region_name)
        stats = compute_drought_statistics(mask)
        return mask, mask_rgb, stats, ndvi, mask_rgba
