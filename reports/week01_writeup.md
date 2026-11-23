# 🌿 TerraSpectra — Week 1 Research & Engineering Write-Up
**Project:** TerraSpectra — Intelligent Crop Disease Detection & Forecasting Using Hyperspectral Imaging  
**Period:** November 12, 2026 – November 17, 2026 (Milestones: Day 1 to Day 3)  
**Author:** Rohit (ML Engineering & Data Pipeline Lead)  
**Working Branch:** `Rohit-EDA-Analysis`  
**Status:** Completed & Validated  

---

## 1. Executive Summary & Week 1 Objectives

During **Week 1**, the TerraSpectra project established the foundational data engineering, sensor radiometry, biophysical feature extraction, and classical machine learning baseline infrastructure. Hyperspectral remote sensing provides contiguous narrow-band sampling (10 nm FWHM) spanning the Visible (VIS, 400–700 nm), Near-Infrared (NIR, 700–1300 nm), and Shortwave-Infrared (SWIR, 1300–2500 nm) spectral regions, capturing plant biochemical shifts invisible to standard RGB or broadband multispectral sensors.

The core technical milestones achieved in Week 1 include:
1. **Sensor & Benchmark Acquisition (Day 1):** Scientific trade-off evaluation comparing spaceborne (EO-1 Hyperion, PRISMA, EnMAP) against airborne (NASA AVIRIS) platforms, establishing AVIRIS Indian Pines ($145 \times 145 \times 200$) and Salinas Valley ($512 \times 217 \times 204$) as the dual benchmark datasets.
2. **Structural Tensor Ingestion & Atmospheric Calibration (Day 2):** Implementation of radiometric float scaling ($\rho \in [0, 1]$), optical mapping across the 4 AVIRIS spectrometer subsystems (A, B, C, D), identification and removal of atmospheric water vapor absorption bands (104–108, 150–163, 220), and engineering the modular 3D `HyperspectralPatchExtractor`.
3. **The Spatial Autocorrelation Leakage Audit (Day 2):** Empirical quantification of Tobler's First Law of Geography in agricultural remote sensing, revealing that naive random pixel splits induce $>88\%$ patch overlap between train and test partitions. Formulation of **Spatial Disjoint Parcel Splitting** via 8-connectivity connected-component analysis (39 isolated parcels in Indian Pines).
4. **Narrowband Biophysical Feature Extraction (Day 3):** Mathematical formulation and spatial rasterization of 8 diagnostic vegetation indices ($\text{NDVI}$, $\text{NDRE}$, $\text{PRI}$, $\text{WBI}$, $\text{MSI}$, $\text{CRI}_1$, $\text{MCARI}$, $\text{SAVI}$), targeting chlorophyll breakdown, red-edge inflection, xanthophyll LUE, and foliar water deficit.
5. **Spectral Angle Mapper (SAM) & Classical ML Benchmark (Day 3):** Full $16 \times 16$ angular distance matrix construction and comparative evaluation of 5 classical algorithms (RF, Extra Trees, SVM-RBF, KNN, SAM).
6. **Empirical Discovery of the Generalization Cliff (Day 3):** Proving that naive random splitting yields illusory test accuracies ($\text{OA} \approx 87.7\%$), which collapse to $\text{OA} \approx 38.6\%$ (a $-48.7\%$ absolute drop, $>55\%$ relative degradation) under spatial disjoint field parcel testing.

---

## 2. Sensor Benchmarking & Dataset Characterization (Day 1)

### 2.1. Spaceborne vs. Airborne Trade-Off Analysis
A systematic multi-criteria evaluation was conducted across orbital and suborbital platforms:

| Sensor Platform | Type | Calibrated Bands | Spectral Coverage | GSD | SNR | Ground Truth Availability |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **NASA EO-1 Hyperion** | Spaceborne | 198 | 355 – 2577 nm | 30 m | ~50:1 (SWIR) | Requires USDA CDL cross-referencing; high atmospheric noise |
| **ASI PRISMA** | Spaceborne | 230 | 400 – 2505 nm | 30 m | >200:1 (VNIR) | Commercial/restricted tasking; mixed pixel challenges |
| **DLR EnMAP** | Spaceborne | 224 | 420 – 2450 nm | 30 m | >400:1 (VNIR) | Recent orbital deployment; limited agricultural field annotations |
| **AVIRIS Indian Pines** | Airborne | **200** | **400 – 2450 nm** | **20 m** | **>1000:1 (VNIR)** | **16 field-verified crop & tillage ground-truth classes** |
| **AVIRIS Salinas Valley** | Airborne | **204** | **400 – 2500 nm** | **3.7 m** | **>1000:1 (VNIR)** | **16 classes; multi-week phenology & weed competition** |

**Selection Rationale:**
- **AVIRIS Indian Pines (`Indian_pines_corrected.mat`):** Benchmark for staple crop discrimination (Corn, Soybeans, Wheat) under varied conservation tillage systems (no-till, min-till, clean-till).
- **AVIRIS Salinas Valley (`Salinas_corrected.mat`):** High spatial resolution ($3.7\text{ m}$) benchmark capturing weed competition (`Brocoli_green_weeds`) and sequential temporal phenology (`Lettuce_romaine` at 4, 5, 6, and 7 weeks), mimicking progressive pathogen colonization.

---

## 3. Structural Tensor Calibration & The Spatial Leakage Audit (Day 2)

### 3.1. Tensor Representation & Radiometric Normalization
Calibrated hyperspectral scenes are ingested as 3rd-order tensors $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$. AVIRIS Digital Numbers ($DN$) are scaled by $10^4$:
$$\rho_{i, j, b} = \mathrm{clip}\left(\frac{\mathcal{X}_{i, j, b}}{10000.0}, 0.0, 1.0\right)$$
- Indian Pines: $145 \times 145 \times 200$ channels, 4,205,000 voxels, 8.02 MB memory footprint (C-contiguous).
- Salinas Valley: $512 \times 217 \times 204$ channels, 22,665,216 voxels, 43.23 MB memory footprint.

### 3.2. AVIRIS Spectrometer Subsystem Decomposition
AVIRIS isolates electromagnetic radiation via 4 separate optical grating spectrometers:
- **Spectrometer A (Bands 1–32, 400–700 nm):** Visible light; chlorophyll $a/b$, carotenoids, anthocyanins.
- **Spectrometer B (Bands 33–96, 700–1300 nm):** NIR plateau; mesophyll structural scattering, Leaf Area Index (LAI).
- **Spectrometer C (Bands 97–149, 1300–1900 nm):** SWIR-1; foliar liquid water and protein absorption.
- **Spectrometer D (Bands 150–200, 1900–2450 nm):** SWIR-2; cellulose, lignin, and structural carbohydrate overtones.
- **Water Vapor Absorption Removal:** Bands 104–108 ($1353–1393\text{ nm}$), Bands 150–163 ($1811–1939\text{ nm}$), and Band 220 ($2450\text{ nm}$) exhibit near-zero atmospheric transmission and were excised.

### 3.3. Spatial Patch Extraction & The Spatial Autocorrelation Leakage Proof
- **Receptive Fields:** The `HyperspectralPatchExtractor` supports kernel window sizes $k \in \{5, 9, 13, 17\}$ with symmetric boundary reflection padding:
  $$p = \lfloor k / 2 \rfloor, \quad \mathcal{P}_{i,j} \in \mathbb{R}^{k \times k \times B}$$
- **Data Leakage Proof:** Under Tobler's First Law ("all things are related, but near things are more related than distant things"), agricultural fields possess high spatial autocorrelation ($r > 0.92$ at distance $d \le 3\text{ pixels}$). 
- When an $80/20$ naive random pixel split is applied to spatial patches of size $9 \times 9$, **$>88\%$ of test patches overlap spatially with training patches**. Models trained under random splits simply memorize spatial coordinates rather than learning generalizable spectral-biochemical signatures.
- **Solution:** Implemented **Disjoint Parcel Splitting**, clustering ground truth pixels into 39 contiguous agricultural fields via 8-connectivity. Entire parcels are isolated into mutually exclusive train and test sets with $0\%$ spatial patch overlap.

---

## 4. Biophysical Indices, SAM Geometry & Classical ML Baselines (Day 3)

### 4.1. Narrowband Biophysical Indices Engine
Hyperspectral resolution allows targeted probing of plant biochemistry:
1. **Normalized Difference Vegetation Index ($\text{NDVI}$):** $(\rho_{800} - \rho_{670})/(\rho_{800} + \rho_{670})$ — General vegetative vigor.
2. **Normalized Difference Red Edge ($\text{NDRE}$):** $(\rho_{790} - \rho_{720})/(\rho_{790} + \rho_{720})$ — Red-edge slope inflection; highly sensitive to early chlorosis where NDVI saturates.
3. **Photochemical Reflectance Index ($\text{PRI}$):** $(\rho_{531} - \rho_{570})/(\rho_{531} + \rho_{570})$ — Xanthophyll cycle de-epoxidation; tracks photosynthetic light-use efficiency.
4. **Water Band Index ($\text{WBI}$):** $\rho_{900} / \rho_{970}$ — Foliar water overtone; early indicator of vascular wilt and drought stress.
5. **Moisture Stress Index ($\text{MSI}$):** $\rho_{1600} / \rho_{820}$ — Leaf desiccation ratio.
6. **Carotenoid Reflectance Index 1 ($\text{CRI}_1$):** $(1/\rho_{510}) - (1/\rho_{550})$ — Carotenoid/chlorophyll ratio under pathogen defense.
7. **Modified Chlorophyll Absorption in Reflectance Index ($\text{MCARI}$):** Soil-adjusted chlorophyll absorption.
8. **Soil-Adjusted Vegetation Index ($\text{SAVI}$):** Canopy vigor attenuating background soil brightness.

### 4.2. Spectral Angle Mapper (SAM) Class Geometry
The spectral angle $\theta(\mathbf{x}, \mathbf{y}) = \arccos\left(\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}\right)$ was computed across all 16 class endmembers. Severe spectral confusion was identified between identical crops under differing tillage regimes:
- `Corn-notill` vs `Corn-mintill`: $\Delta \theta = 0.69^\circ$
- `Soybean-notill` vs `Soybean-mintill`: $\Delta \theta = 0.81^\circ$
This confirmed that purely spectral, non-spatial classifiers cannot reliably differentiate tillage regimes without spatial texture context.

### 4.3. Classical Machine Learning Benchmark & The Generalization Cliff
Five classical algorithms were evaluated across both evaluation regimes:

| Model Architecture | Naive Random Split OA | Disjoint Parcel Split OA | Absolute OA Collapse ($\Delta$) | Relative Degradation (%) | Random Split $\kappa$ | Parcel Split $\kappa$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (100 Trees)** | **87.37%** | **38.63%** | **-48.74%** | **-55.79%** | 0.8552 | 0.2930 |
| **Extra Trees (100 Trees)** | **87.71%** | **38.32%** | **-49.39%** | **-56.31%** | 0.8592 | 0.2903 |
| **k-Nearest Neighbors ($k=5$)** | 77.37% | 33.99% | -43.38% | -56.07% | 0.7409 | 0.2382 |
| **SVM (RBF Kernel, $C=10$)** | 73.37% | 32.89% | -40.48% | -55.17% | 0.6895 | 0.2234 |
| **Spectral Angle Mapper (SAM)**| 51.27% | 15.54% | -35.73% | -69.69% | 0.4509 | 0.0689 |

### 4.4. Random Forest Wavelength Feature Attribution
Gini importance ranking across all 200 channels identified the top physiological diagnostic wavelengths:
- **Red Chlorophyll Absorption Peak:** Band 33 ($671.1\text{ nm}$, Gini $= 0.0412$).
- **Red-Edge Inflection Transition:** Bands 38–41 ($718.3 – 746.5\text{ nm}$, cumulative Gini $= 0.1287$).
- **SWIR Lignin/Cellulose Absorption:** Bands 179–183 ($2234.3 – 2272.4\text{ nm}$, cumulative Gini $= 0.0965$).

---

## 5. Week 1 Deliverables & Repository Artifacts

- **Scripts:**
  - `scripts/download_data.py`: Automated acquisition and integrity validation of raw `.mat` files.
  - `scripts/day01_dataset_verification.py`: Radiometric validation and wavelength mapping.
  - `scripts/day02_structural_inspection_patching.py`: Spectrometer calibration, 3D patch extraction, and disjoint parcel split generation.
  - `scripts/day03_spectral_indices_and_classical_baselines.py`: Biophysical index generation, SAM matrix computation, and classical ML benchmarking.
- **Jupyter Notebooks:**
  - `notebooks/01_dataset_exploration.ipynb`: Exploratory data analysis and class histograms.
  - `notebooks/02_structural_inspection_and_patching.ipynb`: Tensor inspection, water absorption proof, and patch extraction demo.
  - `notebooks/03_spectral_indices_and_classical_baselines.ipynb`: Vegetation index visualization, SAM heatmap, and ML benchmarking.
- **Figures:**
  - `reports/figures/day01_sensor_comparison.png`, `day01_hyperspectral_cube_preview.png`, `day01_indian_pines_classes.png`, `day01_salinas_classes.png`.
  - `reports/figures/day02_spectral_cube_slicing.png`, `day02_spectral_channel_calibration.png`, `day02_spatial_patch_extraction.png`, `day02_spatial_autocorrelation_leakage.png`.
  - `reports/figures/day03_vegetation_indices_spatial_maps.png`, `day03_spectral_angle_mapper_matrix.png`, `day03_classical_ml_benchmark_cliff.png`, `day03_rf_wavelength_importance.png`.
- **Data Logs:**
  - `data/processed/day02_structural_inspection_summary.json`, `day02_spectral_calibration_details.csv`, `day02_patch_extraction_metadata.json`.
  - `data/processed/day03_vegetation_indices_summary.csv`, `day03_spectral_angle_matrix.csv`, `day03_model_benchmark_results.json`, `day03_band_importance_ranking.csv`.
