"""
Google Earth Engine (GEE) Pipeline for DroughtVision AI.
Exact Python 1:1 Translation of PBL Phase 1 GEE Algorithm.
Supports: Mumbai (City + Suburban), Thane, and Raigad.
"""
import ee

DEFAULT_PROJECT = "pbl-sem-v"

def initialize_gee(project_id=DEFAULT_PROJECT):
    """Initializes Earth Engine API."""
    try:
        ee.Initialize(project=project_id)
        return True, f"Google Earth Engine initialized successfully with project '{project_id}'."
    except Exception as e:
        return False, f"GEE initialization notice: {e}"

# GAUL Level2 District Definitions matching user GEE script
DISTRICT_CONFIGS = {
    "Mumbai": {
        "label": "Mumbai (City + Suburban)",
        "center": [19.0760, 72.8777],
        "zoom": 11,
        "description": "Merged Mumbai City and Suburban administrative districts."
    },
    "Thane": {
        "label": "Thane District",
        "center": [19.2183, 72.9781],
        "zoom": 10,
        "description": "Thane administrative district (Konkan Division)."
    },
    "Raigad": {
        "label": "Raigad District",
        "center": [18.5158, 73.1822],
        "zoom": 10,
        "description": "Raigad administrative district (Konkan Division)."
    }
}

def get_district_geometry(district_name):
    """
    Selects district geometry from FAO/GAUL/2015/level2 matching GEE script logic:
    - Mumbai: Merges 'Mumbai city' and 'Mumbai Suburban'
    - Thane: 'Thane'
    - Raigad: 'Raigarh'
    """
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
    
    if district_name == "Mumbai":
        city = gaul.filter(ee.Filter.And(
            ee.Filter.eq("ADM1_NAME", "Maharashtra"),
            ee.Filter.eq("ADM2_NAME", "Mumbai city")
        ))
        suburban = gaul.filter(ee.Filter.And(
            ee.Filter.eq("ADM1_NAME", "Maharashtra"),
            ee.Filter.eq("ADM2_NAME", "Mumbai Suburban")
        ))
        district = city.merge(suburban)
    elif district_name == "Thane":
        district = gaul.filter(ee.Filter.And(
            ee.Filter.eq("ADM1_NAME", "Maharashtra"),
            ee.Filter.eq("ADM2_NAME", "Thane")
        ))
    else:  # Raigad
        district = gaul.filter(ee.Filter.And(
            ee.Filter.eq("ADM1_NAME", "Maharashtra"),
            ee.Filter.eq("ADM2_NAME", "Raigarh")
        ))
        
    return district.geometry()

def make_composite(geometry, start_date, end_date):
    """
    Sentinel-2 Level-2A cloud-masked composite inside season.
    Selects B4 (Red), B8 (NIR), B11 (SWIR), SCL.
    SCL Mask: 4 (Vegetation), 5 (Bare Soil), 7 (Unclassified).
    Returns clipped image with NDVI and NDMI.
    """
    start_str = str(start_date)
    end_str = str(end_date)
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(geometry)
                  .filterDate(start_str, end_str)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
                  .select(["B4", "B8", "B11", "SCL"]))

    def mask_scl(image):
        scl = image.select("SCL")
        mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(7))
        return image.updateMask(mask).select(["B4", "B8", "B11"])

    masked = collection.map(mask_scl)
    composite = masked.median()

    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = composite.normalizedDifference(["B8", "B11"]).rename("NDMI")

    return ee.Image.cat([ndvi, ndmi]).clip(geometry)

def calculate_z(target, mean, std):
    """Calculates Z-scores for NDVI and NDMI."""
    ndvi_std = std.select("NDVI_stdDev").max(ee.Image.constant(0.01))
    ndmi_std = std.select("NDMI_stdDev").max(ee.Image.constant(0.01))

    ndvi_z = (target.select("NDVI")
              .subtract(mean.select("NDVI"))
              .divide(ndvi_std)
              .rename("NDVI_Z"))

    ndmi_z = (target.select("NDMI")
              .subtract(mean.select("NDMI"))
              .divide(ndmi_std)
              .rename("NDMI_Z"))

    return ee.Image.cat([ndvi_z, ndmi_z])

def make_stress(z_image):
    """Computes seasonal stress index from Z-scores."""
    ndvi_stress = (z_image.select("NDVI_Z")
                   .multiply(-1)
                   .max(0)
                   .min(3)
                   .divide(3))

    ndmi_stress = (z_image.select("NDMI_Z")
                   .multiply(-1)
                   .max(0)
                   .min(3)
                   .divide(3))

    return (ndvi_stress.add(ndmi_stress)
            .divide(2)
            .rename("Seasonal_Stress"))

def run_pbl_phase1_drought_analysis(district_name="Mumbai", target_year=2025):
    """
    Executes full GEE Phase 1 Drought Algorithm 1:1 matching user script:
    - 5-year historical baseline (2020-2024) across Pre, Mon, Post seasons.
    - Z-scores & seasonal stress calculation.
    - Weighted Multi-seasonal Drought Score & Persistence classification.
    Returns:
        dict with live MapId tile URLs and quantitative statistical metrics.
    """
    geometry = get_district_geometry(district_name)

    # Historical Baseline (2020-2024)
    pre_images = [make_composite(geometry, f"{y}-02-01", f"{y}-06-01") for y in range(2020, 2025)]
    mon_images = [make_composite(geometry, f"{y}-06-01", f"{y}-10-01") for y in range(2020, 2025)]
    post_images = [make_composite(geometry, f"{y}-10-01", f"{y+1}-02-01") for y in range(2020, 2025)]

    pre_coll = ee.ImageCollection.fromImages(pre_images)
    mon_coll = ee.ImageCollection.fromImages(mon_images)
    post_coll = ee.ImageCollection.fromImages(post_images)

    pre_mean, pre_std = pre_coll.mean(), pre_coll.reduce(ee.Reducer.stdDev())
    mon_mean, mon_std = mon_coll.mean(), mon_coll.reduce(ee.Reducer.stdDev())
    post_mean, post_std = post_coll.mean(), post_coll.reduce(ee.Reducer.stdDev())

    # Target Season (2025)
    target_pre = make_composite(geometry, f"{target_year}-02-01", f"{target_year}-06-01")
    target_mon = make_composite(geometry, f"{target_year}-06-01", f"{target_year}-10-01")
    target_post = make_composite(geometry, f"{target_year}-10-01", f"{target_year+1}-02-01")

    # Z-scores
    pre_z = calculate_z(target_pre, pre_mean, pre_std)
    mon_z = calculate_z(target_mon, mon_mean, mon_std)
    post_z = calculate_z(target_post, post_mean, post_std)

    # Seasonal Stress
    pre_stress = make_stress(pre_z)
    mon_stress = make_stress(mon_z)
    post_stress = make_stress(post_z)

    # Drought Score: 0.50 * Pre + 0.20 * Mon + 0.30 * Post
    drought_score = (pre_stress.multiply(0.50)
                     .add(mon_stress.multiply(0.20))
                     .add(post_stress.multiply(0.30))
                     .rename("Drought_Score"))

    # Persistence
    persistence = (pre_stress.gte(0.167)
                   .add(mon_stress.gte(0.167))
                   .add(post_stress.gte(0.167))
                   .rename("Persistence"))

    # Classification (1 to 5)
    drought_class = (ee.Image(1)
                     .where(drought_score.gte(0.20), 2)
                     .where(drought_score.gte(0.40), 3)
                     .where(drought_score.gte(0.60), 4)
                     .where(drought_score.gte(0.80), 5))

    # Persistence Adjustment
    drought_class = (drought_class
                     .where(persistence.gte(2).And(drought_class.lt(4)), 4)
                     .where(persistence.gte(3), 5)
                     .rename("Drought_Class")
                     .toByte())

    # Map Visualizations
    palette = ["006400", "7CFC00", "FFFF00", "FFA500", "FF0000"]

    # 1. Drought Score Tile URL
    score_map_id = drought_score.getMapId({
        "min": 0.0,
        "max": 1.0,
        "palette": palette
    })
    score_tile_url = score_map_id["tile_fetcher"].url_format

    # 2. Drought Classification Tile URL
    class_map_id = drought_class.getMapId({
        "min": 1,
        "max": 5,
        "palette": palette
    })
    class_tile_url = class_map_id["tile_fetcher"].url_format

    # 3. High Risk Tile URL (Masked)
    high_risk_img = drought_class.gte(4).selfMask()
    high_risk_map_id = high_risk_img.getMapId({
        "palette": ["FF0000"]
    })
    high_risk_tile_url = high_risk_map_id["tile_fetcher"].url_format

    # District Vector Polygon GeoJSON
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
    if district_name == "Mumbai":
        c = gaul.filter(ee.Filter.And(ee.Filter.eq("ADM1_NAME", "Maharashtra"), ee.Filter.eq("ADM2_NAME", "Mumbai city")))
        s = gaul.filter(ee.Filter.And(ee.Filter.eq("ADM1_NAME", "Maharashtra"), ee.Filter.eq("ADM2_NAME", "Mumbai Suburban")))
        dist_fc = c.merge(s)
    elif district_name == "Thane":
        dist_fc = gaul.filter(ee.Filter.And(ee.Filter.eq("ADM1_NAME", "Maharashtra"), ee.Filter.eq("ADM2_NAME", "Thane")))
    else:
        dist_fc = gaul.filter(ee.Filter.And(ee.Filter.eq("ADM1_NAME", "Maharashtra"), ee.Filter.eq("ADM2_NAME", "Raigarh")))

    district_geojson = dist_fc.getInfo()

    # Quantitative Statistics (reduceRegion)
    mean_score = drought_score.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=100, maxPixels=1000000000, bestEffort=True
    ).get("Drought_Score").getInfo()

    persistence_val = persistence.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=100, maxPixels=1000000000, bestEffort=True
    ).get("Persistence").getInfo()

    high_risk_area = ee.Image.pixelArea().updateMask(drought_class.gte(4))
    risk_area_m2 = high_risk_area.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=100, maxPixels=1000000000, bestEffort=True
    ).get("area").getInfo() or 0

    total_area_m2 = ee.Image.pixelArea().reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=100, maxPixels=1000000000, bestEffort=True
    ).get("area").getInfo() or 1

    risk_percent = (risk_area_m2 / total_area_m2) * 100.0
    high_risk_km2 = risk_area_m2 / 1000000.0
    total_area_km2 = total_area_m2 / 1000000.0

    return {
        "district": district_name,
        "score_tile_url": score_tile_url,
        "class_tile_url": class_tile_url,
        "high_risk_tile_url": high_risk_tile_url,
        "district_geojson": district_geojson,
        "drought_score": round(float(mean_score or 0), 4),
        "persistence": round(float(persistence_val or 0), 2),
        "high_risk_km2": round(float(high_risk_km2), 2),
        "total_area_km2": round(float(total_area_km2), 2),
        "high_risk_percent": round(float(risk_percent), 2)
    }

# Fallback presets for compatibility
PRESET_REGIONS = DISTRICT_CONFIGS
