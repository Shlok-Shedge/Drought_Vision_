# 🛰️ DroughtVision AI: Deep Learning Satellite Drought Severity Mapping

> **Project 20**: Satellite Image Analysis using Semantic Segmentation for Real-time Drought Severity Index (DSI) Mapping.  
> *(DroughtVision AI: A Deep Learning Framework for Real-Time Drought Severity Mapping using Satellite Imagery)*

---

## 🌟 Overview
DroughtVision AI is an end-to-end framework combining **Earth Observation (Sentinel-2 multispectral satellite imagery)** with **Deep Learning Semantic Segmentation (Multispectral U-Net)** to generate pixel-level real-time Drought Severity Index (DSI) maps aligned with the U.S. Drought Monitor (USDM) / WMO standards.

Designed with a **hybrid cloud-edge architecture**:
- Heavy multi-epoch model training is handled in the cloud (**Google Colab with Free T4 GPU**).
- Real-time prediction, visualization, and interactive dashboards run efficiently on everyday laptops (**Dell Latitude 7420 / Intel CPU** in `< 85 ms`).

---

## 📁 Project Structure

```
PBL PROJECT/
├── app.py                                   # Streamlit Web Application & Interactive Dashboard
├── requirements.txt                         # Lightweight dependencies for your Dell laptop
├── colab/
│   └── DroughtVision_Model_Training.ipynb   # Complete training notebook for Google Colab (Free GPU)
├── models/                                  # Trained weights directory (e.g. drought_unet.pth)
├── src/
│   ├── model.py                             # PyTorch Multispectral U-Net (4-6 channel input)
│   ├── inference.py                         # Fast CPU-friendly inference engine
│   ├── gee_pipeline.py                      # Google Earth Engine Python API connection
│   └── utils.py                             # Spectral indices (NDVI, NDWI), colormaps, metrics
└── README.md
```

---

## 🚀 Quickstart (Running Locally on Your Laptop)

### 1. Install Dependencies
Open PowerShell or Command Prompt in this folder:
```bash
cd "D:\NAH_NAH\COLLEGE\_SEM_5\PBL PROJECT"
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
```bash
streamlit run app.py
```
Open the provided URL (`http://localhost:8501`) in your browser.

- **Instant Demo Mode**: Use the *"Synthetic Satellite Feed"* to test drought progression interactively with sliders (works 100% offline).
- **Trained Model**: Once you run the Colab notebook, place the exported `drought_unet.pth` inside the `models/` folder.

---

## ☁️ Cloud Model Training (Google Colab)

1. Open [Google Colab](https://colab.research.google.com).
2. Upload `colab/DroughtVision_Model_Training.ipynb`.
3. Set runtime to **GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
4. Run all cells:
   - Trains the **Multispectral U-Net** on satellite bands.
   - Evaluates pixel accuracy and loss curves.
   - Automatically downloads `drought_unet.pth` (~45 MB) to your local machine.
5. Move `drought_unet.pth` into your laptop's `models/` folder.

---

## 🔬 Mathematical Formulas & USDM Drought Scale

### 1. Normalized Difference Vegetation Index (NDVI)
$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

### 2. USDM Drought Severity Index Classes
| Class ID | Code | Severity Description | Typical NDVI Range | Color Representation |
| :--- | :--- | :--- | :--- | :--- |
| **0** | None | No Drought / Healthy Canopy | $> 0.50$ | 🟢 Forest Green (`#2E7D32`) |
| **1** | D0 | Abnormally Dry | $0.38 - 0.50$ | 🟡 Yellow (`#FFEB3B`) |
| **2** | D1 | Moderate Drought | $0.28 - 0.38$ | 🟠 Amber (`#FFB300`) |
| **3** | D2 | Severe Drought | $0.18 - 0.28$ | 🟧 Deep Orange (`#EF6C00`) |
| **4** | D3 | Extreme Drought | $0.10 - 0.18$ | 🔴 Crimson (`#D81B60`) |
| **5** | D4 | Exceptional Drought | $\le 0.10$ | 🟤 Dark Maroon (`#6E0000`) |
