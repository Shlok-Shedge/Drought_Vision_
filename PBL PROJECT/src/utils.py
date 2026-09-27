import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# USDM Drought Severity Index Standard Classes & Colors
DSI_CLASSES = {
    0: {"name": "No Drought (Normal/Wet)", "color": [46, 125, 50],   "hex": "#2E7D32"},
    1: {"name": "D0 - Abnormally Dry",      "color": [255, 235, 59],  "hex": "#FFEB3B"},
    2: {"name": "D1 - Moderate Drought",    "color": [255, 179, 0],   "hex": "#FFB300"},
    3: {"name": "D2 - Severe Drought",      "color": [239, 108, 0],   "hex": "#EF6C00"},
    4: {"name": "D3 - Extreme Drought",     "color": [216, 27, 96],   "hex": "#D81B60"},
    5: {"name": "D4 - Exceptional Drought", "color": [110, 0, 0],     "hex": "#6E0000"}
}

def calculate_ndvi(nir, red):
    """Normalized Difference Vegetation Index (NDVI) = (NIR - Red) / (NIR + Red)"""
    denom = nir + red
    denom = np.where(denom == 0, 1e-6, denom)
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)

def calculate_ndwi(green, nir):
    """Normalized Difference Water Index (NDWI) = (Green - NIR) / (Green + NIR)"""
    denom = green + nir
    denom = np.where(denom == 0, 1e-6, denom)
    ndwi = (green - nir) / denom
    return np.clip(ndwi, -1.0, 1.0)

def calculate_ndmi(nir, swir):
    """Normalized Difference Moisture Index (NDMI) = (NIR - SWIR) / (NIR + SWIR)"""
    denom = nir + swir
    denom = np.where(denom == 0, 1e-6, denom)
    ndmi = (nir - swir) / denom
    return np.clip(ndmi, -1.0, 1.0)

def mask_to_rgb(mask):
    """Convert an integer class segmentation mask [H, W] to an RGB visualization image [H, W, 3]."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, meta in DSI_CLASSES.items():
        rgb[mask == cls_idx] = meta["color"]
    return rgb

def get_district_shape_mask(size=256, region_name=""):
    """
    Creates a binary land/district polygon mask (1 for land, 0 for ocean/outside boundary).
    Matches coastal shapes like Mumbai/Thane in Konkan or inland district shapes.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    if "Konkan" in region_name:
        # Coastal shoreline peninsula (Arabian Sea to the left, Mumbai/Thane land to the right)
        coastline = 0.2 * np.sin(3 * yy) + 0.1 * np.cos(5 * yy) - 0.1
        land_mask = xx > coastline
    elif "Pune" in region_name:
        # Plateau polygon boundary
        r = np.sqrt(xx**2 + (yy * 1.2)**2)
        land_mask = (r <= 0.85) & (yy < 0.8)
    elif "Amravati" in region_name:
        # Vidarbha district polygon boundary
        land_mask = (np.abs(xx) + np.abs(yy) <= 1.3) & (xx > -0.7) & (yy < 0.7)
    else:
        land_mask = np.ones((size, size), dtype=bool)
        
    return land_mask

def mask_to_rgba(mask, region_name="", alpha_val=220):
    """
    Converts segmentation mask to RGBA [H, W, 4] with transparent background/ocean
    matching vector district clipping in Google Earth Engine.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    land_mask = get_district_shape_mask(size=h, region_name=region_name)
    
    for cls_idx, meta in DSI_CLASSES.items():
        cls_pixels = (mask == cls_idx) & land_mask
        rgba[cls_pixels, :3] = meta["color"]
        rgba[cls_pixels, 3] = alpha_val
        
    # Ocean / out-of-boundary pixels are fully transparent (alpha = 0)
    rgba[~land_mask, 3] = 0
    return rgba

def compute_drought_statistics(mask):
    """Computes percentage of area covered by each drought severity category."""
    total_pixels = mask.size
    stats = []
    for cls_idx, meta in DSI_CLASSES.items():
        count = int(np.sum(mask == cls_idx))
        pct = (count / total_pixels) * 100.0
        stats.append({
            "class_id": cls_idx,
            "name": meta["name"],
            "hex": meta["hex"],
            "pixel_count": count,
            "percentage": round(pct, 2)
        })
    return stats

def generate_synthetic_multispectral_tile(size=256, drought_severity=0.5):
    """
    Generates a realistic synthetic multispectral satellite tile [B, G, R, NIR]
    Useful for offline testing on laptop without requiring GEE authentication.
    drought_severity: float between 0.0 (lush, wet) and 1.0 (severe drought)
    """
    np.random.seed(42)
    # Generate spatial terrain gradient
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    xx, yy = np.meshgrid(x, y)
    terrain = np.sin(xx) * np.cos(yy) + 0.3 * np.random.normal(0, 0.2, (size, size))

    # Bands simulation: Blue, Green, Red, NIR (values normalized between 0.0 and 1.0)
    # In healthy vegetation: NIR is high, Red is low
    # In drought: NIR drops significantly, Red and Soil reflectance increase
    dryness_factor = drought_severity + 0.2 * terrain
    dryness_factor = np.clip(dryness_factor, 0.0, 1.0)

    blue = np.clip(0.1 + 0.15 * dryness_factor + np.random.normal(0, 0.02, (size, size)), 0, 1)
    green = np.clip(0.25 - 0.1 * dryness_factor + np.random.normal(0, 0.02, (size, size)), 0, 1)
    red = np.clip(0.15 + 0.35 * dryness_factor + np.random.normal(0, 0.02, (size, size)), 0, 1)
    nir = np.clip(0.8 - 0.6 * dryness_factor + np.random.normal(0, 0.03, (size, size)), 0, 1)

    # Shape: [4, H, W]
    bands = np.stack([blue, green, red, nir], axis=0).astype(np.float32)
    return bands
