# 🌿 TerraSpectra: Hyperspectral Crop Dataset Documentation & Scientific Evaluation
**Project:** TerraSpectra — Intelligent Crop Disease Detection & Forecasting Using Hyperspectral Imaging  
**Role:** ML Engineering & Data Pipeline (Exploratory Data Analysis)  
**Author:** Rohit  
**Working Branch:** `Rohit-EDA-Analysis`  
**Milestone:** Day 1 — Dataset Selection, Source Verification, Metadata Cataloging & Scientific Sensor Justification  
**Last Updated:** November 2026  

---

## 1. Executive Summary & Objective

TerraSpectra focuses on detecting crop stress and potential disease conditions before visible symptoms emerge, leveraging hyperspectral imaging spanning 200+ contiguous spectral channels. Unlike conventional RGB imaging (which only captures broad Red, Green, and Blue bands) or standard multispectral satellites (e.g., Sentinel-2 with 13 discrete bands), hyperspectral imaging captures narrow, contiguous spectral bands across the **Visible (VIS)**, **Near-Infrared (NIR)**, and **Shortwave-Infrared (SWIR)** regions.

This document details the scientific rationale, acquisition, validation, and metadata cataloging of the benchmark datasets selected for the exploratory data analysis and baseline feature pipeline supporting the future **Hybrid 3D-CNN and Vision Transformer (ViT)** architecture.

---

## 2. Sensor Evaluation: Spaceborne vs. Airborne Platforms

Hyperspectral data for precision agriculture originates from either spaceborne satellite platforms or airborne sensor campaigns. A rigorous technical comparison was conducted across the leading candidates:

| Specification Parameter | NASA EO-1 Hyperion (Satellite) | ASI PRISMA (Satellite) | DLR EnMAP (Satellite) | AVIRIS Indian Pines (Airborne Benchmark) | AVIRIS Salinas Valley (Airborne Benchmark) |
|---|---|---|---|---|---|
| **Platform Type** | Spaceborne Satellite | Spaceborne Satellite | Spaceborne Satellite | Airborne (NASA ER-2 Aircraft) | Airborne (NASA ER-2 Aircraft) |
| **Operational Era** | 2000 – 2017 (Archive) | 2019 – Present | 2022 – Present | June 1992 (Gold Standard) | October 1998 (Gold Standard) |
| **Total Spectral Channels** | 242 bands | 239 bands | 242 bands | 220 bands (raw) | 224 bands (raw) |
| **Calibrated Usable Channels** | 198 bands | 230 bands | 224 bands | **200 bands** | **204 bands** |
| **Spectral Coverage** | 355 nm – 2577 nm | 400 nm – 2505 nm | 420 nm – 2450 nm | **400 nm – 2450 nm** | **400 nm – 2500 nm** |
| **Spectral Resolution (FWHM)** | 10 nm | 9 – 12 nm | 6.5 – 10 nm | **10 nm nominal** | **10 nm nominal** |
| **Ground Sample Distance (GSD)** | 30 m | 30 m | 30 m | **20 m** | **3.7 m (High Spatial Detail)** |
| **Swath Width** | 7.5 km | 30 km | 30 km | Variable flight line | Variable flight line |
| **Signal-to-Noise Ratio (SNR)** | ~50:1 (SWIR) to 150:1 (VNIR) | >200:1 (VNIR), >100:1 (SWIR) | >400:1 (VNIR), >150:1 (SWIR) | **>1000:1 (VNIR), >500:1 (SWIR)** | **>1000:1 (VNIR), >500:1 (SWIR)** |
| **Ground-Truth Availability** | USDA CDL required | Point surveys / field campaigns | Field campaigns | **Field-verified agricultural parcel labels** | **Field-verified crop phenology & weed labels** |
| **Suitability for ML Training** | High noise; mixed pixels | High quality, but restricted access | High quality, recent mission | **Benchmark dataset; standardized ground truth** | **Benchmark dataset; growth & senescence stages** |

### Scientific Justification of Selected Datasets:
1. **Primary Dataset: AVIRIS Indian Pines**
   - **Reason:** It represents the universally recognized gold standard in hyperspectral remote sensing of agricultural environments. Captured over agricultural parcels in Indiana, it provides 200 clean, contiguous spectral channels at 20 m GSD with 16 distinct crop and land-cover categories, including major staple crops (Corn, Soybeans, Wheat) subjected to different tillage regimes (no-till, min-till, clean).
2. **Complementary Dataset: AVIRIS Salinas Valley Scene**
   - **Reason:** Salinas provides high spatial resolution (3.7 m GSD) over intensive agricultural fields in California. Crucially, it includes classes representing **senesced vegetation** (`Corn_senesced_green_weeds`), **weed competition** (`Brocoli_green_weeds_1/2`), and **multi-week temporal phenology** (`Lettuce_romaine` at 4, 5, 6, and 7 weeks). These phenological and senescent variations exhibit the exact biochemical degradation patterns (chlorophyll breakdown, water loss, cellular decay) characteristic of early crop stress and pathogen attacks.
3. **Spaceborne Synergy (Hyperion, PRISMA, EnMAP):**
   - The models and feature extraction algorithms developed on the 200-band AVIRIS benchmarks directly transfer to satellite sensors like PRISMA and EnMAP, as both spaceborne platforms operate across the exact same 400–2500 nm domain at 10 nm spectral sampling.

---

## 3. Dataset Characteristics & Spatial/Spectral Metadata

The acquired raw datasets have been stored in `data/raw/` and validated. The extracted metadata is summarized below:

### 3.1. AVIRIS Indian Pines (`Indian_pines_corrected.mat`)
- **Spatial Grid:** 145 lines $\times$ 145 samples (21,025 total pixels).
- **Spectral Depth:** 200 calibrated channels (raw dataset has 220 bands; 20 atmospheric water absorption bands removed).
- **Spatial Resolution:** 20.0 meters per pixel.
- **Physical Area Covered:** Approximately $2.9\text{ km} \times 2.9\text{ km}$ ($8.41\text{ km}^2$).
- **Radiometric Format:** 16-bit unsigned integer (`uint16`).
- **Dynamic Range:** Minimum digital number (DN) = 955, Maximum DN = 9,604, Mean DN = 2,875.4 (reflectance scaled by 10,000).
- **Ground Truth Mask:** 145 $\times$ 145 (`uint8`), comprising 10,249 labeled pixels (48.75% of scene) and 10,776 background/unlabeled pixels (51.25%).

### 3.2. AVIRIS Salinas Valley (`Salinas_corrected.mat`)
- **Spatial Grid:** 512 lines $\times$ 217 samples (111,104 total pixels).
- **Spectral Depth:** 204 calibrated channels (raw dataset has 224 bands; 20 atmospheric absorption bands removed).
- **Spatial Resolution:** 3.7 meters per pixel.
- **Physical Area Covered:** High-density agricultural parcel tracking.
- **Radiometric Format:** 16-bit signed integer (`int16`).
- **Dynamic Range:** Minimum DN = -11, Maximum DN = 9,207, Mean DN = 2,541.3.
- **Ground Truth Mask:** 512 $\times$ 217 (`uint8`), comprising 54,129 labeled pixels (48.72% of scene) and 56,975 background pixels (51.28%).

---

## 4. Spectral Band Architecture & Wavelength Mapping

AVIRIS collects solar reflected radiance using four separate optical spectrometers covering the following contiguous regions:

```
[ VIS: 400 - 700 nm ] ---> [ NIR: 700 - 1100 nm ] ---> [ SWIR-1: 1100 - 1800 nm ] ---> [ SWIR-2: 1900 - 2500 nm ]
       |                            |                              |                             |
Chlorophyll Absorption          Red-Edge & Mesophyll          Atmospheric H2O Absorption     Cellulose, Lignin &
(Blue ~450nm, Red ~660nm)       Reflectance Plateau           & Plant Moisture Bands         Dry Matter Absorption
```

### 4.1. Atmospheric Absorption Bands (Why They Must Be Filtered)
Earth's atmosphere absorbs solar radiation at specific resonant molecular frequencies:
- **Water Vapor Absorption Zone 1 ($\lambda \in [1360, 1420\text{ nm}]$):** Raw Bands 104–108 in Indian Pines. In this window, transmittance drops to near zero, yielding noise.
- **Water Vapor Absorption Zone 2 ($\lambda \in [1800, 1950\text{ nm}]$):** Raw Bands 150–163 in Indian Pines. Severe atmospheric attenuation.
- **Detector Edge / Low SNR ($\lambda \sim 2450 - 2500\text{ nm}$):** Band 220.

The 200-band corrected dataset retains only physically meaningful reflectance channels, preventing machine learning models from overfitting to atmospheric noise artifacts.

---

## 5. Ground-Truth Class Distribution & Analysis

### 5.1. Indian Pines Agricultural Classes
The scene encompasses 16 verified classes:

| Class ID | Class Name | Pixel Count | % of Total Scene | % of Labeled Samples | Agricultural Relevance |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | Background / Unlabeled | 10,776 | 51.25% | — | Unclassified areas, roads, field margins |
| **1** | Alfalfa | 46 | 0.22% | 0.45% | Forage legume, nitrogen fixing |
| **2** | Corn-notill | 1,428 | 6.79% | 13.93% | Maize with crop residue cover (stubble) |
| **3** | Corn-mintill | 830 | 3.95% | 8.10% | Maize with reduced tillage |
| **4** | Corn | 237 | 1.13% | 2.31% | Maize with conventional clean tillage |
| **5** | Grass-pasture | 483 | 2.30% | 4.71% | Permanent pasture grass |
| **6** | Grass-trees | 730 | 3.47% | 7.12% | Mixed pasture and arboreal cover |
| **7** | Grass-pasture-mowed | 28 | 0.13% | 0.27% | Mechanically cut pasture (biomass reduction) |
| **8** | Hay-windrowed | 478 | 2.27% | 4.66% | Cut hay drying in field rows |
| **9** | Oats | 20 | 0.10% | 0.20% | Cereal grain (highly limited sample size) |
| **10** | Soybean-notill | 972 | 4.62% | 9.48% | Soybeans planted directly into residue |
| **11** | Soybean-mintill | 2,455 | 11.68% | 23.95% | Soybeans under conservation tillage (dominant class) |
| **12** | Soybean-clean | 593 | 2.82% | 5.79% | Soybeans under conventional tilled soil |
| **13** | Wheat | 205 | 0.98% | 2.00% | Winter wheat canopy |
| **14** | Woods | 1,265 | 6.02% | 12.34% | Dense forest canopy |
| **15** | Buildings-Grass-Trees-Drives | 386 | 1.84% | 3.77% | Mixed urban/farmstead vegetation |
| **16** | Stone-Steel-Towers | 95 | 0.45% | 0.93% | Anthropogenic structures (man-made non-veg) |

### 5.2. Salinas Valley Classes (Phenology & Senescence Focus)
The Salinas dataset provides direct proxies for crop stress and growth stages:
- **Growth Stage Progression:** `Lettuce_romaine_4wk` (1,068 px), `5wk` (1,927 px), `6wk` (3,959 px), `7wk` (1,079 px) illustrate vegetative canopy expansion and leaf area index (LAI) progression.
- **Senescence & Weed Stress:** `Corn_senesced_green_weeds` (3,278 px) captures the spectral combination of declining chlorophyll content (senescence) coupled with weed competition.
- **Soil & Residue Baseline:** `Fallow`, `Fallow_rough_plow`, `Fallow_smooth`, and `Stubble` represent soil background baselines necessary for isolating pure canopy spectral responses.

---

## 6. Critical Observations & Challenges for ML Engineering

1. **Extreme Class Imbalance:**
   - In Indian Pines, sample counts range from 2,455 pixels (`Soybean-mintill`) down to 20 pixels (`Oats`), a ratio of **122.75 : 1**.
   - Standard random sampling will starve minority classes. The ML pipeline must employ **stratified spatial sampling**, **focal loss**, or **class-weighted cross-entropy**.
2. **Spatial Autocorrelation & Data Leakage:**
   - Agricultural fields exist in contiguous spatial clusters. If individual pixels from the same field are randomly partitioned into training and testing sets, the model will suffer from data leakage. Spatial patch-based cross-validation is essential.
3. **High Spectral Redundancy:**
   - Contiguous 10 nm bands exhibit substantial collinearity ($r > 0.95$ across adjacent channels). Dimensionality reduction (PCA) and band selection (Days 10 & 16) will be crucial for the 3D-CNN / ViT model.
4. **Distinction Between Stress Signatures and Disease Ground Truth:**
   - *Scientific Disclaimer:* The benchmark labels classify crop categories, tillage regimes, and phenological stages, not specific pathogen strains (e.g., *Fusarium*, *Puccinia*). However, the biochemical symptoms of disease (chlorophyll loss, moisture deficit, tissue necrosis) mirror the spectral differences observable between senesced crops, weed-infested parcels, and healthy canopies.

---

## 7. Generated Project Artifacts (Day 1)

The following verified artifacts were produced and registered in the project workspace:
- **Processed Metadata:**
  - `data/processed/indian_pines_metadata.json`: Full technical profile of Indian Pines.
  - `data/processed/indian_pines_wavelengths.csv`: 220-band wavelength centers, spectral regions, and water-vapor flags.
  - `data/processed/indian_pines_class_distribution.csv`: Pixel counts and proportions for all 16 classes.
  - `data/processed/salinas_metadata.json`: Technical profile of Salinas Valley.
  - `data/processed/salinas_class_distribution.csv`: Class distribution across phenological stages.
- **High-Resolution Figures (`reports/figures/`):**
  - `day01_sensor_comparison.png`: Comparison of spaceborne (Hyperion, PRISMA, EnMAP) and airborne sensors.
  - `day01_indian_pines_classes.png`: Class sample frequency distribution.
  - `day01_salinas_classes.png`: Salinas phenology and crop condition distribution.
  - `day01_hyperspectral_cube_preview.png`: 4-panel false-color composite and ground-truth spatial maps.
- **Notebooks:**
  - `notebooks/01_dataset_exploration.ipynb`: End-to-end reproducible Jupyter Notebook for dataset exploration.
