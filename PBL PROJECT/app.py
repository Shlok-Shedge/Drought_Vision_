import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import src.utils
import src.model
import src.inference
import src.gee_pipeline
importlib.reload(src.utils)
importlib.reload(src.model)
importlib.reload(src.inference)
importlib.reload(src.gee_pipeline)

from src.inference import DroughtInferenceEngine
from src.gee_pipeline import (
    initialize_gee, 
    run_pbl_phase1_drought_analysis,
    DISTRICT_CONFIGS
)
from src.utils import DSI_CLASSES, generate_synthetic_multispectral_tile

import folium
try:
    from streamlit_folium import st_folium
    HAS_FOLIUM_UI = True
except ImportError:
    HAS_FOLIUM_UI = False

st.set_page_config(
    page_title="DroughtVision AI | PBL Phase 1 Drought Mapping",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title & Introduction
st.title("🛰️ DroughtVision AI: PBL Phase 1 Drought Mapping")
st.caption("Real-Time Multi-Seasonal Satellite Drought Classification (Sentinel-2 Harmonized Baseline 2020–2024 vs 2025 Target)")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration & Data Source")

data_source = st.sidebar.radio(
    "Select Data Source:",
    [
        "🌐 Google Earth Engine (Live GEE Engine)",
        "⚡ Offline Synthetic Simulator"
    ]
)

st.sidebar.markdown("---")
selected_district = st.sidebar.selectbox(
    "Select Target District:",
    ["Mumbai", "Thane", "Raigad"]
)

dist_cfg = DISTRICT_CONFIGS.get(selected_district, {})
st.sidebar.caption(f"📍 **District Info**: {dist_cfg.get('description', '')}")

st.sidebar.markdown("---")
target_year = st.sidebar.number_input("Target Analysis Year:", min_value=2024, max_value=2025, value=2025)

# Main Processing Section
st.markdown(f"### 📊 GEE Phase 1 Analysis: {selected_district} District ({int(target_year)})")

gee_ok, gee_msg = initialize_gee()

if "Google Earth Engine" in data_source and gee_ok:
    with st.spinner(f"Executing GEE Multi-Seasonal Baseline & Z-Score Engine for {selected_district}..."):
        try:
            results = run_pbl_phase1_drought_analysis(district_name=selected_district, target_year=int(target_year))
            st.success(f"✅ GEE Phase 1 Execution Successful for **{selected_district}** (Baseline 2020–2024 vs {target_year})")
        except Exception as e:
            st.error(f"GEE Execution Error: {e}")
            results = None
else:
    if not gee_ok:
        st.warning(f"GEE Connection Notice: {gee_msg}. Switch to Google Earth Engine once authenticated.")
    results = None

# Quantitative Results Display
if results:
    st.subheader("📈 Quantitative District Results")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Overall Drought Score", f"{results['drought_score']:.4f}", delta="Target 2025")
    kpi2.metric("High Risk Area", f"{results['high_risk_km2']} km²", f"{results['high_risk_percent']}% of District")
    kpi3.metric("Stress Persistence", f"{results['persistence']} / 3.0", "Seasons Active")
    kpi4.metric("Total District Area", f"{results['total_area_km2']} km²", selected_district)

    # Interactive Map Section
    st.markdown("---")
    st.subheader(f"🗺️ Google Earth Engine Interactive Map: {selected_district}")
    st.caption("Live GEE Tile Layers: Drought Classification, Drought Score, and High Risk Mask clipped to official district polygon.")

    if HAS_FOLIUM_UI:
        col_ctrl, col_map = st.columns([1, 4])
        with col_ctrl:
            st.write("##### Layer Controls")
            opacity = st.slider("Layer Opacity:", 0.0, 1.0, 0.85, 0.05)
            show_sat = st.checkbox("Satellite Basemap", value=True)
            active_layer = st.radio(
                "Primary Map Display:",
                ["Drought Classification (Classes 1-5)", "Drought Score (0.0 - 1.0)", "High Risk Areas Mask (Classes 4-5)"]
            )

        with col_map:
            center_coords = dist_cfg.get("center", [19.0760, 72.8777])
            zoom_lvl = dist_cfg.get("zoom", 10)

            m = folium.Map(
                location=center_coords,
                zoom_start=zoom_lvl,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" if show_sat else "OpenStreetMap",
                attr="Esri World Imagery" if show_sat else "OpenStreetMap"
            )

            # Draw Official District GeoJSON Vector Polygon Boundary
            if results.get("district_geojson"):
                folium.GeoJson(
                    results["district_geojson"],
                    name=f"{selected_district} District Boundary",
                    style_function=lambda x: {
                        'fillColor': 'transparent',
                        'color': '#FFFFFF',
                        'weight': 3,
                        'dashArray': '3, 3'
                    },
                    tooltip=f"<b>Official Boundary: {selected_district}</b>"
                ).add_to(m)

            # Add Live GEE Tile Layers
            if "Classification" in active_layer:
                folium.TileLayer(
                    tiles=results["class_tile_url"],
                    attr="Google Earth Engine",
                    name=f"{selected_district} Drought Classification",
                    overlay=True,
                    opacity=opacity
                ).add_to(m)
            elif "Score" in active_layer:
                folium.TileLayer(
                    tiles=results["score_tile_url"],
                    attr="Google Earth Engine",
                    name=f"{selected_district} Drought Score",
                    overlay=True,
                    opacity=opacity
                ).add_to(m)
            else:
                folium.TileLayer(
                    tiles=results["high_risk_tile_url"],
                    attr="Google Earth Engine",
                    name=f"{selected_district} High Risk Areas",
                    overlay=True,
                    opacity=opacity
                ).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=530, returned_objects=[])

# Classification Legend & Methodology Overview
st.markdown("---")
col_leg, col_exp = st.columns([1, 2])

with col_leg:
    st.write("##### Classification Legend")
    legend_items = [
        {"name": "Class 1: Low / Normal", "hex": "#006400"},
        {"name": "Class 2: Mild Stress", "hex": "#7CFC00"},
        {"name": "Class 3: Moderate Drought", "hex": "#FFFF00"},
        {"name": "Class 4: High Risk (Severe)", "hex": "#FFA500"},
        {"name": "Class 5: Very High Risk (Extreme)", "hex": "#FF0000"}
    ]
    for item in legend_items:
        st.markdown(
            f"<div style='display:flex; align-items:center; margin-bottom:8px;'>"
            f"<span style='background:{item['hex']}; width:18px; height:18px; display:inline-block; border-radius:4px; margin-right:10px;'></span>"
            f"<span style='font-size:0.95rem; font-weight:600;'>{item['name']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

with col_exp:
    with st.expander("📖 PBL Phase 1 Algorithm & Methodology Overview", expanded=True):
        st.markdown(f"""
        **1. District Geometry**: Clipped strictly to `{selected_district}` via `FAO/GAUL/2015/level2`.  
        **2. Satellite Baseline**: Sentinel-2 Level-2A (`COPERNICUS/S2_SR_HARMONIZED`), SCL Cloud Masking (4, 5, 7).  
        **3. Historical Baseline (2020–2024)**: Pre-monsoon (Feb-Jun), Monsoon (Jun-Oct), Post-monsoon (Oct-Feb).  
        **4. Indices & Z-Scores**: $\\text{{NDVI}} = \\frac{{\\text{{NIR}} - \\text{{Red}}}}{{\\text{{NIR}} + \\text{{Red}}}}$, $\\text{{NDMI}} = \\frac{{\\text{{NIR}} - \\text{{SWIR}}}}{{\\text{{NIR}} + \\text{{SWIR}}}}$.  
        **5. Drought Score Formula**:
        $$\\text{{Drought Score}} = 0.50 \\times \\text{{PreStress}} + 0.20 \\times \\text{{MonStress}} + 0.30 \\times \\text{{PostStress}}$$
        """)
