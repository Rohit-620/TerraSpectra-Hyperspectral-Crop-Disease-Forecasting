# 🌿 TerraSpectra: Narrowband Biophysical Indices, Spectral Angle Mapping & Classical ML Generalization Benchmark
**Project:** TerraSpectra — Intelligent Crop Disease Detection & Forecasting Using Hyperspectral Imaging  
**Role:** ML Engineering & Hyperspectral Remote Sensing Pipeline  
**Author:** Rohit  
**Working Branch:** `Rohit-EDA-Analysis`  
**Milestone:** Day 3 — Spectral Signature Characterization, Biophysical Indices & Classical ML Generalization Cliff  
**Date:** November 17, 2026  

---

## 1. Executive Summary & Day 3 Research Scope

Building upon the dataset calibration, spectrometer mapping, and 3D patch extraction established in **Day 1** and **Day 2**, the core objective of **Day 3** is to rigorously characterize the biochemical spectral signatures of crops, extract specialized narrowband biophysical stress indices, establish physics-based and statistical classification baselines, and empirically quantify the **"Generalization Cliff"** in agricultural remote sensing.

Hyperspectral cubes possess continuous 10 nm spectral bands spanning the 400–2450 nm range. While traditional multispectral instruments (e.g., Landsat, Sentinel-2) only evaluate broad spectral averages, hyperspectral resolution permits targeted physiological probing of foliar pigment concentrations, mesophyll cell integrity, and canopy water content.

This research milestone delivers:
1. **Narrowband Biophysical & Crop Stress Indices Engine:** Mathematical formulation and spatial mapping of 8 specialized narrow-band vegetation indices ($\text{NDVI}$, $\text{NDRE}$, $\text{PRI}$, $\text{WBI}$, $\text{MSI}$, $\text{CRI}_1$, $\text{MCARI}$, $\text{SAVI}$) across Indian Pines and Salinas Valley.
2. **Spectral Angle Mapper (SAM) Analysis & Endmember Geometry:** Rigorous calculation of unit-normalized class endmembers, construction of the full $16 \times 16$ pairwise SAM angular distance matrix, and identification of extreme spectral confusion clusters ($< 0.7^\circ$ angular separation between tillage regimes).
3. **Rigorous Classical ML Benchmarking across Spatial Regimes:** Comparative evaluation of 5 diverse algorithms (Random Forest, Extra Trees, SVM with RBF kernel, $k$-Nearest Neighbors, and Spectral Angle Mapper) under both **Naive Random Split** (with spatial autocorrelation leakage) and **Spatial Disjoint Parcel Split** (zero spatial patch leakage).
4. **Empirical Proof of the Generalization Cliff:** Detailed mathematical and empirical demonstration of the performance drop ($\Delta \text{OA} \approx 48.7\%$, relative collapse $> 55\%$) when models transition from naive random splitting to unseen disjoint field parcels.
5. **Random Forest Wavelength Feature Attribution:** Gini importance ranking of all 200 calibrated spectral channels, mapping dominant diagnostic peaks to chlorophyll absorption ($671\text{ nm}$), red-edge inflection ($718 - 746\text{ nm}$), and SWIR cellulose/lignin absorption ($2234 - 2272\text{ nm}$).

---

## 2. Mathematical Formulation of Narrowband Biophysical Indices

### 2.1. Biophysical Motivation & Biochemical Mechanisms
Unlike broad-band sensors, hyperspectral imaging isolates narrow absorption troughs corresponding to specific molecular transitions and pigment complexes:
- **Chlorophyll $a$ and $b$:** Primary photosynthetic pigments absorbing strongly in the blue ($430 - 460\text{ nm}$) and red ($660 - 680\text{ nm}$) regions. In early disease or nutrient stress, chlorophyll breaks down, triggering foliar chlorosis.
- **Red-Edge Transition ($700 - 740\text{ nm}$):** The boundary between strong red chlorophyll absorption and intense near-infrared internal leaf scattering. The red-edge slope and inflection point shift toward shorter wavelengths ("blue shift") during chlorosis and tissue necrosis.
- **Xanthophyll Cycle Pigments ($531\text{ nm}$):** Under excess solar irradiance or photosynthetic stress, plants convert violaxanthin to antheraxanthin and zeaxanthin for non-photochemical quenching (NPQ), measurable via minute reflectance shifts at $531\text{ nm}$.
- **Foliar Liquid Water Absorption ($970\text{ nm}, 1450\text{ nm}, 1940\text{ nm}$):** Vibrational overtones of fundamental O-H stretching modes in leaf mesophyll water.

### 2.2. Mathematical Definitions of Extracted Indices

Let $\rho_\lambda$ denote calibrated surface reflectance at center wavelength $\lambda$ (in nanometers):

| Index Acronym | Formal Mathematical Definition | Primary Biophysical Mechanism | Diagnostic Agricultural Utility |
|:---|:---:|:---|:---|
| **NDVI** | $\frac{\rho_{800} - \rho_{670}}{\rho_{800} + \rho_{670}}$ | Chlorophyll red absorption vs mesophyll NIR scattering | Fractional canopy cover, green biomass, broad crop vigor |
| **NDRE** | $\frac{\rho_{790} - \rho_{720}}{\rho_{790} + \rho_{720}}$ | Red-edge slope inflection sensitivity | Early chlorosis detection; remains sensitive where NDVI saturates |
| **PRI** | $\frac{\rho_{531} - \rho_{570}}{\rho_{531} + \rho_{570}}$ | Xanthophyll cycle pigment epoxidation state | Diurnal photosynthetic stress, light-use efficiency (LUE) |
| **WBI** | $\frac{\rho_{900}}{\rho_{970}}$ | Foliar liquid water absorption overtone at 970 nm | Canopy water status, cell turgor pressure, drought/vascular wilt |
| **MSI** | $\frac{\rho_{1600}}{\rho_{820}}$ | SWIR water absorption relative to NIR plateau | Foliar desiccation and leaf moisture deficit |
| **CRI1** | $\frac{1}{\rho_{510}} - \frac{1}{\rho_{550}}$ | Carotenoid to chlorophyll absorption ratio | Premature senescence, oxidative stress, viral/fungal defense |
| **MCARI** | $\left[(\rho_{700} - \rho_{670}) - 0.2 (\rho_{700} - \rho_{550})\right] \frac{\rho_{700}}{\rho_{670}}$ | Soil-adjusted chlorophyll absorption | Chlorophyll quantification minimizing background soil reflectance |
| **SAVI** | $\frac{(1 + L)(\rho_{800} - \rho_{670})}{\rho_{800} + \rho_{670} + L}, \quad L=0.5$ | Soil background brightness attenuation | Canopy vigor in early growth stages and row crops |

### 2.3. Statistical Profiling across Agricultural Classes (Indian Pines)
Inspection of `data/processed/day03_vegetation_indices_summary.csv` reveals distinct biophysical separation:

| Class ID | Agricultural Class Name | Sample Pixels | $\text{NDVI}$ (Mean $\pm$ Std) | $\text{NDRE}$ (Mean $\pm$ Std) | $\text{PRI}$ (Mean $\pm$ Std) | $\text{WBI}$ (Mean $\pm$ Std) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | Alfalfa | 46 | $0.3277 \pm 0.046$ | $0.1221 \pm 0.027$ | $0.0470 \pm 0.010$ | $1.0834 \pm 0.015$ |
| 2 | Corn-notill | 1,428 | $0.1172 \pm 0.021$ | $0.0726 \pm 0.013$ | $0.0278 \pm 0.009$ | $1.1030 \pm 0.021$ |
| 3 | Corn-mintill | 830 | $0.1273 \pm 0.020$ | $0.0803 \pm 0.013$ | $0.0290 \pm 0.009$ | $1.1123 \pm 0.022$ |
| 4 | Corn | 237 | $0.1840 \pm 0.041$ | $0.1150 \pm 0.024$ | $0.0373 \pm 0.011$ | $1.1369 \pm 0.027$ |
| 5 | Grass-pasture | 483 | $0.4559 \pm 0.049$ | $0.2649 \pm 0.033$ | $0.0712 \pm 0.015$ | $1.1832 \pm 0.028$ |
| 6 | Grass-trees | 730 | $0.3711 \pm 0.057$ | $0.1841 \pm 0.034$ | $0.1082 \pm 0.019$ | $1.1418 \pm 0.028$ |
| 8 | Hay-windrowed | 478 | $0.2918 \pm 0.045$ | $0.1145 \pm 0.022$ | $0.0459 \pm 0.011$ | $1.0738 \pm 0.017$ |
| 10 | Soybean-notill | 972 | $0.1077 \pm 0.017$ | $0.0618 \pm 0.011$ | $0.0287 \pm 0.008$ | $1.1399 \pm 0.020$ |
| 11 | Soybean-mintill | 2,455 | $0.1111 \pm 0.024$ | $0.0681 \pm 0.016$ | $0.0269 \pm 0.010$ | $1.0954 \pm 0.026$ |
| 12 | Soybean-clean | 593 | $0.1286 \pm 0.019$ | $0.0773 \pm 0.014$ | $0.0311 \pm 0.008$ | $1.1162 \pm 0.021$ |
| 13 | Wheat | 205 | $0.3481 \pm 0.037$ | $0.2639 \pm 0.024$ | $0.1535 \pm 0.017$ | $1.1934 \pm 0.021$ |
| 14 | Woods | 1,265 | $0.5286 \pm 0.042$ | $0.3553 \pm 0.035$ | $0.0934 \pm 0.014$ | $1.1921 \pm 0.023$ |
| 16 | Stone-Steel-Towers | 93 | $0.0082 \pm 0.024$ | $0.0130 \pm 0.016$ | $0.0269 \pm 0.013$ | $1.2121 \pm 0.026$ |

**Key Biochemical Findings:**
1. **Canopy Closure & Biomass Stratification:** Dense vegetative canopies (`Woods`, `Grass-pasture`) exhibit high NDVI ($0.45 - 0.53$) and high NDRE ($0.26 - 0.36$), reflecting healthy mesophyll structure and strong chlorophyll absorption.
2. **Crop Residue & Tillage Suppression:** The major grain crops (`Corn-notill`, `Soybean-notill`, `Soybean-mintill`) exhibit low NDVI ($0.10 - 0.13$) because the scene was acquired in early autumn when mature crops had begun drying or were surrounded by exposed tilled soil and crop residues.
3. **Non-Vegetated Baseline:** Anthropogenic structures (`Stone-Steel-Towers`) show near-zero NDVI ($0.0082$) and NDRE ($0.0130$), providing a clean non-vegetative reference.

---

## 3. Spectral Angle Mapper (SAM) Analysis & Endmember Geometry

### 3.1. Mathematical Formulation of SAM
Spectral Angle Mapper (SAM) is a physics-based spectral classification technique based on projective geometry in $B$-dimensional space. Given a test spectral signature vector $\mathbf{x} \in \mathbb{R}^B$ and a reference class endmember vector $\mathbf{e}_c \in \mathbb{R}^B$, the spectral angle $\theta(\mathbf{x}, \mathbf{e}_c)$ is defined as:

$$\theta(\mathbf{x}, \mathbf{e}_c) = \arccos\left(\frac{\mathbf{x} \cdot \mathbf{e}_c}{\|\mathbf{x}\|_2 \|\mathbf{e}_c\|_2}\right)$$

Where:
- $\theta \in [0, \pi]$ represents the angular distance in radians (or degrees).
- Smaller angles indicate higher spectral similarity.
- **Illumination Invariance:** Because the dot product is divided by the product of vector magnitudes, scalar multipliers (such as topographical shadowing, cloud shadows, or variations in solar irradiance) cancel out:
  $$\theta(\alpha \mathbf{x}, \beta \mathbf{e}_c) = \arccos\left(\frac{\alpha \beta (\mathbf{x} \cdot \mathbf{e}_c)}{\alpha \|\mathbf{x}\|_2 \beta \|\mathbf{e}_c\|_2}\right) = \theta(\mathbf{x}, \mathbf{e}_c) \quad \forall \alpha, \beta > 0$$
  This makes SAM an essential remote sensing baseline for evaluating spectral curve shape independent of albedo magnitude.

### 3.2. Endmember Generation & Pairwise Spectral Angle Matrix
For each of the 16 classes in Indian Pines, a reference endmember $\mathbf{e}_c$ was constructed by normalizing individual pixel spectra to unit length and computing the class centroid:
$$\mathbf{e}_c = \frac{\sum_{i \in \mathcal{S}_c} \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2}}{\left\|\sum_{i \in \mathcal{S}_c} \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2}\right\|_2}$$

The full $16 \times 16$ pairwise angular distance matrix was computed (Figure 2: `reports/figures/day03_spectral_angle_mapper_matrix.png`).

### 3.3. Empirical Identification of Severe Spectral Confusion Clusters
Inspection of `data/processed/day03_spectral_angle_matrix.csv` reveals why pixel-wise spectral classifiers face profound difficulties:

| Rank | Class 1 | Class 2 | Spectral Angle ($\theta$) | Cosine Similarity | Agricultural Context |
|:---:|:---|:---|:---:|:---:|:---|
| **1** | **Corn-notill** | **Corn-mintill** | **$0.699^\circ$** ($0.0122\text{ rad}$) | **$0.99993$** | Same crop type; slight difference in soil residue fraction |
| **2** | **Soybean-notill** | **Soybean-mintill** | **$0.846^\circ$** ($0.0148\text{ rad}$) | **$0.99989$** | Same cultivar; conservation vs minimum tillage residue |
| **3** | **Soybean-mintill** | **Soybean-clean** | **$0.932^\circ$** ($0.0163\text{ rad}$) | **$0.99987$** | Minimum tillage vs clean tilled soil background |
| **4** | **Corn-notill** | **Corn** | **$1.121^\circ$** ($0.0196\text{ rad}$) | **$0.99981$** | Same species under varying canopy maturities |
| **5** | **Alfalfa** | **Oats** | **$1.385^\circ$** ($0.0242\text{ rad}$) | **$0.99971$** | Broadleaf forage vs grain canopy |
| ... | ... | ... | ... | ... | ... |
| **120** | **Woods** | **Stone-Steel-Towers** | **$22.569^\circ$** ($0.3939\text{ rad}$) | **$0.92342$** | Dense biological canopy vs anthropogenic metal/concrete |

**Critical Insight:**  
The angular distance between tillage variants of the same crop (`Corn-notill` vs `Corn-mintill`, `Soybean-notill` vs `Soybean-mintill`) is **less than $1^\circ$ in 200-dimensional space**! Their spectral reflectance curves are virtually identical across all channels. Differentiating between these classes purely on 1D spectral vectors requires distinguishing noise-level differences. To reliably separate them, models **must leverage 2D spatial texture and parcel-level context**, which motivates the 3D-CNN and Transformer architectures planned for Day 4.

---

## 4. Dual Spatial Evaluation Regimes: Naive Random Split vs Disjoint Parcel Split

To evaluate classical machine learning models rigorously, two contrasting experimental partitions were established:

```
+-------------------------------------------------------------------------------------------------------+
|                                    EXPERIMENTAL PARTITION REGIMES                                     |
+---------------------------------------------+---------------------------------------------------------+
| Regime A: Naive Random Split (80/20)        | Regime B: Spatial Disjoint Parcel Split                 |
+---------------------------------------------+---------------------------------------------------------+
| - Pixels randomly sampled across the scene. | - Connected component labeling segments scene into      |
| - Stratified by class.                      |   43 discrete, contiguous field parcels.                |
| - High spatial autocorrelation:             | - Stratified parcel allocation reserves entire fields   |
|   Test pixels are immediate spatial         |   for testing.                                          |
|   neighbors of training pixels.             | - Zero spatial voxel overlap between train and test.    |
| - Over 60% test patch contamination.        | - True test of generalization to unvisited farm fields. |
+---------------------------------------------+---------------------------------------------------------+
```

### Partition Sample Breakdown:
- **Total Labeled Samples:** 10,249 pixels across 16 classes.
- **Naive Random Split:** Train = 8,199 pixels ($80.0\%$), Test = 2,050 pixels ($20.0\%$).
- **Disjoint Parcel Split:** Train = 7,707 pixels ($75.2\%$ across 34 parcels), Test = 2,542 pixels ($24.8\%$ across 9 parcels).

---

## 5. Classical Machine Learning Benchmark & The Generalization Cliff

Five classical classification algorithms were trained and evaluated across both spatial regimes:
1. **Random Forest (RF):** 100 decision trees with Gini impurity splitting.
2. **Extra Trees Classifier (ET):** 100 extremely randomized trees.
3. **Support Vector Machine (SVM-RBF):** Radial Basis Function kernel with cost parameter $C = 10.0$ and $\gamma = \text{'scale'}$.
4. **$k$-Nearest Neighbors ($k$-NN):** Euclidean distance metric with $k = 5$.
5. **Spectral Angle Mapper (SAM):** Minimum spectral angle to reference class endmembers.

### 5.1. Comprehensive Benchmark Results Table

Results from `data/processed/day03_model_benchmark_results.json` and Figure 3 (`reports/figures/day03_classical_ml_benchmark_cliff.png`):

| Algorithm | Regime A: Random Split OA (%) | Regime A: Random Split $\kappa$ | Regime B: Parcel Split OA (%) | Regime B: Parcel Split $\kappa$ | Absolute OA Drop ($\Delta \text{OA}$) | Relative OA Drop (%) | Fit Time (s) | Inference Latency (ms/px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (100 Trees)** | **87.37%** | **0.8552** | **38.63%** | **0.2930** | **-48.74%** | **-55.79%** | 3.32s | 0.046 ms |
| **Extra Trees (100 Trees)** | **87.71%** | **0.8592** | **38.32%** | **0.2903** | **-49.39%** | **-56.31%** | 1.15s | 0.049 ms |
| **SVM (RBF Kernel, C=10)** | 73.37% | 0.6895 | 32.89% | 0.2234 | -40.48% | -55.17% | 9.42s | 0.852 ms |
| **$k$-Nearest Neighbors (k=5)** | 77.37% | 0.7409 | 33.99% | 0.2382 | -43.38% | -56.07% | 0.02s | 0.584 ms |
| **Spectral Angle Mapper (SAM)**| 51.27% | 0.4509 | 15.54% | 0.0689 | -35.73% | -69.69% | 0.01s | 0.003 ms |

### 5.2. Analysis of the "Generalization Cliff"
The empirical benchmark demonstrates a profound phenomenon in agricultural machine learning:

1. **The Random Split Illusion:**
   - Under naive random splitting, tree ensembles achieve near **$88\%$ overall accuracy** and Cohen's Kappa of **$0.86$**, creating the impression that 1D pixel classification is solved.
   - However, as proven mathematically on Day 2, adjacent pixels in the same parcel share identical soil background, moisture content, illumination angles, and cultivation histories (cosine similarity $> 0.985$). The model is not learning generalizable crop signatures; it is memorizing the specific spectral coordinates of those particular parcels.
2. **The Collapse on Disjoint Parcels:**
   - When deployed on unvisited parcels under Disjoint Parcel Splitting, Random Forest accuracy plunges from **$87.37\%$ to $38.63\%$**, a relative collapse of **$55.79\%$**.
   - Cohen's Kappa collapses from **$0.8552$ (almost perfect agreement)** down to **$0.2930$ (fair/poor agreement)**.
   - All five algorithms exhibit identical relative degradation ($\sim 55\% - 70\%$).
3. **Scientific Implication for TerraSpectra:**
   - A crop disease detection model that claims $95\%$ accuracy on random test pixels will fail when deployed on a neighboring farm.
   - Overcoming this Generalization Cliff requires:
     - **3D Spatial-Spectral Deep Learning:** Combining local spatial texture with spectral absorption features (Day 4).
     - **Biophysical Index Normalization:** Leveraging ratio-based indices ($\text{NDRE}$, $\text{PRI}$) that are invariant to absolute radiometric scaling.
     - **Spatial Domain Adaptation:** Training on parcel-stratified splits to encourage invariant representation learning.

---

## 6. Spectral Band Importance & Biochemical Wavelength Attribution

### 6.1. Random Forest Gini Feature Attribution
To determine which physical wavelengths provide the strongest discriminatory signal for crop classification, Gini feature importances were extracted across all 200 calibrated channels (Figure 4: `reports/figures/day03_rf_wavelength_importance.png`).

### 6.2. Top 15 Diagnostic Absorption Wavelengths
From `data/processed/day03_band_importance_ranking.csv`:

| Rank | Band Index (1-based) | Wavelength ($\lambda$) | Optical Subsystem | Gini Importance | Cumulative Importance | Biochemical & Physiological Mechanism |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **Band 27** | **643.38 nm** | VIS Red | **0.0163** | 1.63% | Chlorophyll-$b$ absorption shoulder; photosynthetic light reaction |
| **2** | **Band 30** | **671.46 nm** | VIS Red | **0.0159** | 3.23% | **Strong Chlorophyll-$a$ absorption peak**; primary photosynthetic driver |
| **3** | **Band 182** | **2272.15 nm** | SWIR-2 | **0.0156** | 4.79% | **Cellulose, lignin, and crop dry residue absorption** |
| **4** | **Band 178** | **2234.70 nm** | SWIR-2 | **0.0149** | 6.28% | Lignin C-H overtone; key for separating no-till from tilled soil |
| **5** | **Band 179** | **2244.06 nm** | SWIR-2 | **0.0135** | 7.63% | Plant structural carbon and stalk residue spectral signature |
| **6** | **Band 32** | **690.18 nm** | VIS Red | **0.0119** | 8.82% | Chlorophyll absorption boundary leading into red-edge transition |
| **7** | **Band 181** | **2262.79 nm** | SWIR-2 | **0.0117** | 9.99% | Cellulose absorption trough in decomposing crop biomass |
| **8** | **Band 149** | **1963.24 nm** | SWIR-2 | **0.0115** | 11.15% | Boundary of major atmospheric water vapor absorption window |
| **9** | **Band 100** | **1326.71 nm** | SWIR-1 | **0.0110** | 12.25% | Foliar liquid water absorption transition; cell turgor indicator |
| **10** | **Band 154** | **2010.05 nm** | SWIR-2 | **0.0110** | 13.35% | Nitrogen and protein N-H stretching overtones in canopy leaves |
| **11** | **Band 180** | **2253.43 nm** | SWIR-2 | **0.0108** | 14.43% | Secondary cellulose/lignin absorption feature |
| **12** | **Band 31** | **680.82 nm** | VIS Red | **0.0104** | 15.47% | Chlorophyll-$a$ fluorescence excitation and red edge onset |
| **13** | **Band 29** | **662.10 nm** | VIS Red | **0.0102** | 16.49% | Red chlorophyll absorption dip; photosystem II activity |
| **14** | **Band 35** | **718.26 nm** | NIR / Red-Edge | **0.0099** | 17.48% | **Red-Edge inflection point**; critical for early chlorosis detection |
| **15** | **Band 175** | **2206.63 nm** | SWIR-2 | **0.0097** | 18.45% | Clay mineral hydroxyl (Al-OH) and dry matter absorption |

**Attribution Analysis:**
- The top 15 bands contribute over **$18.45\%$** of the total decision tree impurity reduction.
- Two primary physical regimes dominate crop discrimination:
  1. **VIS Red & Red-Edge ($640 - 720\text{ nm}$):** Captures chlorophyll content and photosynthetic capacity, separating healthy green crops from senesced vegetation.
  2. **SWIR-2 ($2200 - 2280\text{ nm}$):** Captures lignin, cellulose, and plant residue, providing the discriminatory power to separate no-till, minimum-till, and clean-tilled fields.

---

## 7. Generated Project Artifacts (Day 3)

All code, data, figures, and notebooks for Day 3 have been generated and validated:

### 7.1. Executable Scripts
- `scripts/day03_spectral_indices_and_classical_baselines.py`: Production end-to-end pipeline computing 8 biophysical indices, SAM matrices, dataset partitioning, classical ML benchmarking, RF band ranking, and publication figure generation.
- `scripts/generate_day03_notebook.py`: Automated reproducible notebook generator.

### 7.2. Processed Data & Metadata (`data/processed/`)
- `data/processed/day03_vegetation_indices_summary.csv`: Class-by-class statistical summary (mean, std, median, 5th, 95th percentiles) for NDVI, NDRE, PRI, WBI, MSI, CRI1, MCARI, and SAVI across 16 agricultural classes.
- `data/processed/day03_spectral_angle_matrix.csv`: Complete $16 \times 16$ pairwise SAM angular distance table (256 pairs) with angles in degrees and radians, plus cosine similarities.
- `data/processed/day03_model_benchmark_results.json`: Comprehensive metrics (OA, AA, Cohen's Kappa, Macro F1, Weighted F1, fit times, inference latency, per-class metrics, and Generalization Cliff deltas) across all 5 models and both splitting regimes.
- `data/processed/day03_band_importance_ranking.csv`: Complete ranking of all 200 spectral bands by Random Forest Gini importance with spectral domains and biochemical mechanism annotations.

### 7.3. Publication Figures (`reports/figures/`)
- `day03_vegetation_indices_spatial_maps.png`: Multi-panel spatial heatmaps of NDVI, NDRE, PRI, and WBI for Indian Pines and Salinas Valley.
- `day03_spectral_angle_mapper_matrix.png`: $16 \times 16$ pairwise SAM distance heatmap with numeric confusion overlays and reference endmember spectral curves.
- `day03_classical_ml_benchmark_cliff.png`: Side-by-side comparative bar plots of Overall Accuracy, Average Accuracy, and Cohen's Kappa illustrating the Generalization Cliff.
- `day03_rf_wavelength_importance.png`: 200-band feature importance profile with AVIRIS spectrometer domains shaded and top diagnostic peaks labeled.

### 7.4. Interactive Jupyter Notebook
- `notebooks/03_spectral_indices_and_classical_baselines.ipynb`: 20-cell reproducible analysis notebook with mathematical formulations, code cells, and interactive tables/figures.

---

## 8. Strategic Direction & Day 4 Roadmap

With the biophysical indices established, spectral confusion quantified, and the Generalization Cliff proven on classical models, the technical foundation for deep learning is set:

- **Day 4:** Spatial-Spectral Deep Learning Architectures (1D-CNN vs 2D-CNN vs 3D-CNN):
  - Ingestion of the $9 \times 9 \times B$ 3D spatial-spectral patches generated on Day 2.
  - Designing a modular PyTorch 3D-CNN that simultaneously learns spatial texture (field rows, parcel homogeneity) and spectral absorption depth.
  - Training under Disjoint Parcel Splitting to directly overcome the Generalization Cliff and establish robust agricultural field generalization.
