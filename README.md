# 🌿 TerraSpectra: Intelligent Crop Disease Detection & Forecasting Using Hyperspectral Imaging

## 🔍 About the Project

**TerraSpectra** is an AI-driven agricultural research project focused on leveraging hyperspectral imaging and machine learning to analyze crop health and identify potential disease conditions.

Unlike conventional RGB imaging, hyperspectral imaging captures detailed spectral information across multiple wavelengths. These spectral signatures can reveal subtle variations in plant characteristics that may be associated with disease or stress.

TerraSpectra aims to transform this spectral information into meaningful insights through advanced data processing, feature extraction, and predictive modeling, contributing to smarter and more efficient agricultural monitoring.

## 🎯 Key Research Goals

- Investigate hyperspectral signatures of healthy and diseased crops.
- Identify significant wavelengths associated with crop health conditions.
- Develop efficient spectral preprocessing and feature selection techniques.
- Build intelligent models for crop disease identification.
- Explore early disease indicators and predictive forecasting.
- Evaluate model performance for reliable agricultural analysis.

## 🧪 Proposed Approach

The development process is organized into the following stages:

1. **Hyperspectral Data Acquisition:** Collect relevant crop datasets containing spectral information and disease labels.
2. **Spectral Data Preparation:** Perform data cleaning, normalization, noise reduction, and preprocessing.
3. **Spectral Investigation:** Examine wavelength-dependent patterns and variations in crop samples.
4. **Feature Engineering:** Extract and select informative spectral characteristics for model development.
5. **Intelligent Model Development:** Experiment with machine learning and deep learning architectures.
6. **Disease Identification:** Analyze model predictions to distinguish healthy and diseased crop conditions.
7. **Predictive Analysis:** Investigate disease progression and future risk using suitable temporal and environmental information.
8. **Performance Assessment:** Evaluate results using relevant statistical and machine learning metrics.

## 💻 Technical Stack

| Area | Tools & Technologies |
|---|---|
| Programming | Python |
| Data Analysis | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / PyTorch |
| Image Processing | OpenCV, Spectral Python |
| Visualization | Matplotlib, Seaborn |
| Experimentation | Jupyter Notebook, Google Colab |
| Collaboration | Git, GitHub |

*Technology selection may evolve throughout the research and implementation phases.*

## 📅 8-Day Research & Engineering Milestones

| Day | Milestone | Key Deliverables & Research Scope | Branch / Artifacts |
|:---:|:---|:---|:---|
| **Day 1** | **Dataset Pipeline & Sensor Benchmark** | Spaceborne vs. airborne trade-offs, AVIRIS Indian Pines & Salinas validation, band metadata cataloging. | [docs/dataset_documentation.md](file:///d:/TerraSpectra2026/docs/dataset_documentation.md)<br>[notebooks/01_dataset_exploration.ipynb](file:///d:/TerraSpectra2026/notebooks/01_dataset_exploration.ipynb) |
| **Day 2** | **Radiometric Calibration & Spatial Leakage** | Float scaling ($\rho \in [0, 1]$), water absorption excision, 3D patch extraction, empirical proof of Tobler's spatial leakage ($>88\%$), 39-parcel disjoint splitting. | [docs/day02_structural_inspection_and_patching.md](file:///d:/TerraSpectra2026/docs/day02_structural_inspection_and_patching.md)<br>[notebooks/02_structural_inspection_and_patching.ipynb](file:///d:/TerraSpectra2026/notebooks/02_structural_inspection_and_patching.ipynb) |
| **Day 3** | **Biophysical Indices & Classical Generalization Cliff** | 8 narrowband indices ($\text{NDVI}, \text{NDRE}, \text{PRI}, \text{PSRI}, \text{NDWI}$), SAM angular geometry, discovery of classical Generalization Cliff ($-48.7\%$ OA drop). | [docs/day03_spectral_indices_and_classical_baselines.md](file:///d:/TerraSpectra2026/docs/day03_spectral_indices_and_classical_baselines.md)<br>[reports/week01_writeup.md](file:///d:/TerraSpectra2026/reports/week01_writeup.md) |
| **Day 4** | **Spatial-Spectral Deep Learning Suite** | 1D-CNN, 2D-CNN, 3D-CNN, and HybridSN (3D-2D CNN) on PCA-30 ($99.25\%$ variance), proof of deep boundary overfitting ($-63.45\%$ collapse), full-scene inference. | [docs/day04_spatial_spectral_deep_learning.md](file:///d:/TerraSpectra2026/docs/day04_spatial_spectral_deep_learning.md)<br>[notebooks/04_deep_learning_spatial_spectral.ipynb](file:///d:/TerraSpectra2026/notebooks/04_deep_learning_spatial_spectral.ipynb) |
| **Day 5** | **Transformers & Foundation Self-Supervision** | Hyperspectral Masked Autoencoders (HS-MAE, $70\%$ masking), SpectralFormer ViT, overcoming the cliff ($63.89\%$ parcel OA), Salinas $\to$ Indian Pines CORAL domain transfer. | [docs/day05_transformers_and_self_supervised_learning.md](file:///d:/TerraSpectra2026/docs/day05_transformers_and_self_supervised_learning.md)<br>[notebooks/05_transformers_and_self_supervised_learning.ipynb](file:///d:/TerraSpectra2026/notebooks/05_transformers_and_self_supervised_learning.ipynb) |
| **Day 6** | **Pathology Staging & Epidemiological PDE** | Biophysical Composite Disease Severity Index (DSI), pre-visual warning ($\text{ROC-AUC} = 0.9750$), 2D Reaction-Diffusion PDE microclimate risk forecasting ($t = 0, 3, 7, 14\text{ days}$). | [docs/day06_crop_disease_detection_and_forecasting.md](file:///d:/TerraSpectra2026/docs/day06_crop_disease_detection_and_forecasting.md)<br>[reports/week02_writeup.md](file:///d:/TerraSpectra2026/reports/week02_writeup.md) |
| **Day 7** | **Explainable AI, Precision DSS & Production Deployment** | Integrated Gradients spectral attribution, TreeSHAP biophysical importance, Economic Injury Level ($EIL$), Variable-Rate Application (VRA) achieving **$62.4\%$ chemical reduction**, grand 7-day benchmark synthesis. | [docs/day07_decision_support_xai_and_deployment.md](file:///d:/TerraSpectra2026/docs/day07_decision_support_xai_and_deployment.md)<br>[reports/capstone_research_writeup.md](file:///d:/TerraSpectra2026/reports/capstone_research_writeup.md) |
| **Day 8** | **Edge-AI Quantization, Sensor Robustness & Autonomous UAV Planning** | Dynamic INT8 Quantization ($43.1\%$ memory reduction, $<0.08\%$ accuracy drop), AWGN & atmospheric haze stress testing, proof of biophysical index ratio invariance, Boustrophedon UAV flight mission generator ($841\text{ ha}$, $70.8\text{ Wh}$), real-time edge microservice & GeoJSON prescription export. | [docs/day08_edge_optimization_and_mission_planning.md](file:///d:/TerraSpectra2026/docs/day08_edge_optimization_and_mission_planning.md)<br>[notebooks/08_edge_optimization_and_mission_planning.ipynb](file:///d:/TerraSpectra2026/notebooks/08_edge_optimization_and_mission_planning.ipynb) |

## 📈 Research Deliverables

The project aims to develop:

- A hyperspectral data analysis and preprocessing pipeline.
- Spectral feature extraction and wavelength analysis methods.
- Intelligent crop disease classification models.
- Experimental early disease identification techniques.
- Disease forecasting approaches where suitable data is available.
- Comparative model evaluation and visual analytics.

## 🌾 Potential Use Cases

- Precision Agriculture
- Intelligent Crop Monitoring
- Plant Disease Identification
- Agricultural Remote Sensing
- Smart Farming Solutions
- Sustainable Agriculture
- Data-Driven Farm Management

## 🚀 Future Development

Future enhancements may include drone-based hyperspectral data acquisition, multi-crop disease analysis, spectral-spatial deep learning, integration of environmental parameters, explainable AI, and real-time crop health monitoring systems.

The long-term vision is to explore an intelligent platform capable of transforming hyperspectral observations into actionable agricultural insights.

## 👨‍💻 Collaborative Development

TerraSpectra follows a collaborative development approach using GitHub for source control, project coordination, and contribution management.

Development responsibilities are divided across different modules, including:

- Hyperspectral Data Processing
- Spectral Feature Engineering
- Machine Learning & Deep Learning
- Disease Classification & Forecasting
- Model Evaluation & Visualization

Each contributor works through an independent development branch, with changes integrated into the main branch through Pull Requests and code reviews.

## 🌍 Project Vision

To bridge hyperspectral imaging and artificial intelligence for intelligent crop health assessment, early disease identification, and predictive agricultural monitoring.

**TerraSpectra — Seeing Beyond the Visible, Understanding Crop Health.** 🌱
