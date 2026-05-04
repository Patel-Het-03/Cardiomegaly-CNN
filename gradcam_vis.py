import numpy as np
import cv2

def overlay_cam_on_image(rgb_image: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    """
    rgb_image: HxWx3 uint8
    cam: HxW float in [0,1]
    returns: overlayed HxWx3 uint8
    """
    h, w = rgb_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * rgb_image + alpha * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay