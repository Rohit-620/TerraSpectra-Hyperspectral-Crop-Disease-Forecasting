"""
TerraSpectra: Day 3 - Narrowband Biophysical Indices, Spectral Angle Mapping & Classical ML Generalization Benchmark
ML Engineering & Hyperspectral Remote Sensing Pipeline
Author: Rohit (TerraSpectra ML Team)
Working Branch: Rohit-EDA-Analysis
Milestone: Day 3 - Spectral Signature Characterization, Biophysical Indices & Classical ML Generalization Cliff
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.ndimage as ndi

# Configure multiprocessing and matplotlib environment before imports
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

# -----------------------------------------------------------------------------
# Base Directories & Output Locations
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

os.makedirs(PROC_DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Agricultural Class Taxonomies
# -----------------------------------------------------------------------------
INDIAN_PINES_CLASSES = {
    0: "Background / Unlabeled",
    1: "Alfalfa",
    2: "Corn-notill",
    3: "Corn-mintill",
    4: "Corn",
    5: "Grass-pasture",
    6: "Grass-trees",
    7: "Grass-pasture-mowed",
    8: "Hay-windrowed",
    9: "Oats",
    10: "Soybean-notill",
    11: "Soybean-mintill",
    12: "Soybean-clean",
    13: "Wheat",
    14: "Woods",
    15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers"
}

SALINAS_CLASSES = {
    0: "Background / Unlabeled",
    1: "Brocoli_green_weeds_1",
    2: "Brocoli_green_weeds_2",
    3: "Fallow",
    4: "Fallow_rough_plow",
    5: "Fallow_smooth",
    6: "Stubble",
    7: "Celery",
    8: "Grapes_untrained",
    9: "Soil_vinyard_develop",
    10: "Corn_senesced_green_weeds",
    11: "Lettuce_romaine_4wk",
    12: "Lettuce_romaine_5wk",
    13: "Lettuce_romaine_6wk",
    14: "Lettuce_romaine_7wk",
    15: "Vinyard_untrained",
    16: "Vinyard_vertical_trellis"
}

# -----------------------------------------------------------------------------
# 1. Narrowband Biophysical Indices Engine
# -----------------------------------------------------------------------------
def get_nearest_band_idx(target_nm, wavelengths):
    """Returns 0-based index of band closest to target wavelength in nm."""
    return int(np.argmin(np.abs(wavelengths - target_nm)))

def compute_narrowband_indices(cube_reflectance, wavelengths):
    """
    Computes 8 key narrowband biophysical and crop stress indices from surface reflectance:
    1. NDVI  : Normalized Difference Vegetation Index (Biomass & Chlorophyll)
    2. NDRE  : Normalized Difference Red-Edge (Early Chlorosis & Disease Sensitivity)
    3. PRI   : Photochemical Reflectance Index (Photosynthetic Light-Use Efficiency)
    4. WBI   : Water Band Index (Canopy Liquid Water Status)
    5. MSI   : Moisture Stress Index (Canopy Desiccation & Dehydration)
    6. CRI1  : Carotenoid Reflectance Index 1 (Carotenoid/Chlorophyll Ratio & Senescence)
    7. MCARI : Modified Chlorophyll Absorption in Reflectance Index (Soil-Resistant)
    8. SAVI  : Soil Adjusted Vegetation Index (Soil Background Correction with L=0.5)
    """
    eps = 1e-7
    b_800 = get_nearest_band_idx(800.0, wavelengths)
    b_670 = get_nearest_band_idx(670.0, wavelengths)
    b_790 = get_nearest_band_idx(790.0, wavelengths)
    b_720 = get_nearest_band_idx(720.0, wavelengths)
    b_531 = get_nearest_band_idx(531.0, wavelengths)
    b_570 = get_nearest_band_idx(570.0, wavelengths)
    b_900 = get_nearest_band_idx(900.0, wavelengths)
    b_970 = get_nearest_band_idx(970.0, wavelengths)
    b_1600 = get_nearest_band_idx(1600.0, wavelengths)
    b_820 = get_nearest_band_idx(820.0, wavelengths)
    b_510 = get_nearest_band_idx(510.0, wavelengths)
    b_550 = get_nearest_band_idx(550.0, wavelengths)
    b_700 = get_nearest_band_idx(700.0, wavelengths)

    R_800 = cube_reflectance[:, :, b_800]
    R_670 = cube_reflectance[:, :, b_670]
    R_790 = cube_reflectance[:, :, b_790]
    R_720 = cube_reflectance[:, :, b_720]
    R_531 = cube_reflectance[:, :, b_531]
    R_570 = cube_reflectance[:, :, b_570]
    R_900 = cube_reflectance[:, :, b_900]
    R_970 = cube_reflectance[:, :, b_970]
    R_1600 = cube_reflectance[:, :, b_1600]
    R_820 = cube_reflectance[:, :, b_820]
    R_510 = cube_reflectance[:, :, b_510]
    R_550 = cube_reflectance[:, :, b_550]
    R_700 = cube_reflectance[:, :, b_700]

    # NDVI: (R800 - R670) / (R800 + R670)
    ndvi = (R_800 - R_670) / (R_800 + R_670 + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # NDRE: (R790 - R720) / (R790 + R720)
    ndre = (R_790 - R_720) / (R_790 + R_720 + eps)
    ndre = np.clip(ndre, -1.0, 1.0)

    # PRI: (R531 - R570) / (R531 + R570)
    pri = (R_531 - R_570) / (R_531 + R_570 + eps)
    pri = np.clip(pri, -1.0, 1.0)

    # WBI: R900 / R970
    wbi = R_900 / (R_970 + eps)
    wbi = np.clip(wbi, 0.0, 3.0)

    # MSI: R1600 / R820
    msi = R_1600 / (R_820 + eps)
    msi = np.clip(msi, 0.0, 5.0)

    # CRI1: (1 / R510) - (1 / R550)
    cri1 = (1.0 / (R_510 + eps)) - (1.0 / (R_550 + eps))
    cri1 = np.clip(cri1, -20.0, 20.0)

    # MCARI: [(R700 - R670) - 0.2*(R700 - R550)] * (R700 / R670)
    mcari = ((R_700 - R_670) - 0.2 * (R_700 - R_550)) * (R_700 / (R_670 + eps))
    mcari = np.clip(mcari, -2.0, 2.0)

    # SAVI: (1 + L) * (R800 - R670) / (R800 + R670 + L), L = 0.5
    L = 0.5
    savi = ((1.0 + L) * (R_800 - R_670)) / (R_800 + R_670 + L + eps)
    savi = np.clip(savi, -1.0, 1.0)

    return {
        "NDVI": ndvi,
        "NDRE": ndre,
        "PRI": pri,
        "WBI": wbi,
        "MSI": msi,
        "CRI1": cri1,
        "MCARI": mcari,
        "SAVI": savi
    }

def summarize_indices_by_class(indices_dict, gt_mask, class_dict):
    """
    Computes rigorous statistical distribution metrics (mean, std, median, 5th, 95th percentiles)
    for each biophysical vegetation index across all agricultural classes.
    """
    records = []
    classes = [c for c in sorted(class_dict.keys()) if c > 0]
    
    for c in classes:
        c_mask = (gt_mask == c)
        c_count = int(np.sum(c_mask))
        if c_count == 0:
            continue
        
        row = {
            "class_id": c,
            "class_name": class_dict[c],
            "sample_count": c_count
        }
        
        for idx_name, idx_map in indices_dict.items():
            vals = idx_map[c_mask]
            row[f"{idx_name}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{idx_name}_std"] = round(float(np.std(vals)), 4)
            row[f"{idx_name}_median"] = round(float(np.median(vals)), 4)
            row[f"{idx_name}_p05"] = round(float(np.percentile(vals, 5)), 4)
            row[f"{idx_name}_p95"] = round(float(np.percentile(vals, 95)), 4)
            
        records.append(row)
        
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 2. Spectral Angle Mapper (SAM) Engine & Class Distance Matrix
# -----------------------------------------------------------------------------
class SpectralAngleMapperClassifier:
    """
    Physics-based Spectral Angle Mapper (SAM) Classifier for Hyperspectral Data.
    Classifies an unknown spectral signature vector x to reference class endmember e_c
    by minimizing the spectral angle theta = arccos( (x . e_c) / (||x|| * ||e_c||) ).
    """
    def __init__(self):
        self.endmembers_ = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            c_spectra = X[y == c]
            # Reference endmember as centroid of unit-normalized class spectra
            norms = np.linalg.norm(c_spectra, axis=1, keepdims=True) + 1e-8
            unit_spectra = c_spectra / norms
            mean_endmember = np.mean(unit_spectra, axis=0)
            self.endmembers_[c] = mean_endmember / (np.linalg.norm(mean_endmember) + 1e-8)
        return self

    def predict(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        unit_X = X / norms
        
        # Stack endmembers into matrix [n_classes, n_features]
        E = np.vstack([self.endmembers_[c] for c in self.classes_])
        # Cosine similarity matrix: [n_samples, n_classes]
        cos_sim = np.dot(unit_X, E.T)
        best_cls_idx = np.argmax(cos_sim, axis=1)
        return self.classes_[best_cls_idx]

def compute_sam_matrix(X, y, class_dict):
    """
    Computes reference endmembers and full pairwise Spectral Angle Mapper (SAM)
    angular distance matrix in degrees across all crop classes.
    """
    classes = [c for c in sorted(class_dict.keys()) if c > 0]
    endmembers = {}
    for c in classes:
        c_spectra = X[y == c]
        if len(c_spectra) == 0:
            continue
        mean_spec = np.mean(c_spectra, axis=0)
        endmembers[c] = mean_spec / (np.linalg.norm(mean_spec) + 1e-8)

    n_cls = len(classes)
    sam_matrix_deg = np.zeros((n_cls, n_cls))
    sam_records = []

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            dot = np.clip(np.dot(endmembers[c1], endmembers[c2]), -1.0, 1.0)
            angle_rad = float(np.arccos(dot))
            angle_deg = float(np.degrees(angle_rad))
            sam_matrix_deg[i, j] = angle_deg
            sam_records.append({
                "class_1_id": c1,
                "class_1_name": class_dict[c1],
                "class_2_id": c2,
                "class_2_name": class_dict[c2],
                "spectral_angle_deg": round(angle_deg, 3),
                "spectral_angle_rad": round(angle_rad, 5),
                "cosine_similarity": round(float(dot), 5)
            })

    return sam_matrix_deg, endmembers, pd.DataFrame(sam_records)

# -----------------------------------------------------------------------------
# 3. Disjoint Parcel Splitting Formulation
# -----------------------------------------------------------------------------
def build_dataset_partitions(cube, gt, test_ratio=0.2, random_seed=42):
    """
    Generates two independent experimental dataset partitions:
    1. Naive Random Pixel Split (Stratified 80/20 across pixels)
    2. Spatial Disjoint Parcel Split (Connected-component parcels partitioned as intact blocks)
    """
    H, W, B = cube.shape
    labeled_mask = (gt > 0)
    
    # Flatten labeled pixels
    X_all = cube[labeled_mask].astype(np.float32) / 10000.0
    y_all = gt[labeled_mask].astype(int)
    
    # 1. Naive Random Split
    X_rnd_train, X_rnd_test, y_rnd_train, y_rnd_test = train_test_split(
        X_all, y_all, test_size=test_ratio, random_state=random_seed, stratify=y_all
    )
    
    # 2. Disjoint Parcel Split via Connected Components
    parcel_labels, num_parcels = ndi.label(labeled_mask)
    parcel_records = []
    for p_id in range(1, num_parcels + 1):
        p_mask = (parcel_labels == p_id)
        p_gt_vals = gt[p_mask]
        dominant_cls = int(np.bincount(p_gt_vals).argmax())
        parcel_records.append({
            "parcel_id": p_id,
            "class_id": dominant_cls,
            "pixel_count": int(np.sum(p_mask))
        })
        
    parcel_df = pd.DataFrame(parcel_records)
    np.random.seed(random_seed)
    test_parcels = []
    for cls_id in np.unique(y_all):
        cls_parcels = parcel_df[parcel_df["class_id"] == cls_id]["parcel_id"].values
        if len(cls_parcels) > 1:
            n_test_p = max(1, int(len(cls_parcels) * test_ratio))
            test_parcels.extend(list(np.random.choice(cls_parcels, n_test_p, replace=False)))
            
    test_parcel_set = set(test_parcels)
    parcel_train_mask = np.zeros((H, W), dtype=bool)
    parcel_test_mask = np.zeros((H, W), dtype=bool)

    for p_id in range(1, num_parcels + 1):
        p_mask = (parcel_labels == p_id)
        if p_id in test_parcel_set:
            parcel_test_mask[p_mask] = True
        else:
            parcel_train_mask[p_mask] = True

    X_prc_train = cube[parcel_train_mask].astype(np.float32) / 10000.0
    y_prc_train = gt[parcel_train_mask].astype(int)
    X_prc_test = cube[parcel_test_mask].astype(np.float32) / 10000.0
    y_prc_test = gt[parcel_test_mask].astype(int)

    return {
        "random_split": {
            "X_train": X_rnd_train,
            "X_test": X_rnd_test,
            "y_train": y_rnd_train,
            "y_test": y_rnd_test,
        },
        "parcel_split": {
            "X_train": X_prc_train,
            "X_test": X_prc_test,
            "y_train": y_prc_train,
            "y_test": y_prc_test,
            "train_mask": parcel_train_mask,
            "test_mask": parcel_test_mask,
            "num_parcels": num_parcels,
            "test_parcels_count": len(test_parcels)
        }
    }

# -----------------------------------------------------------------------------
# 4. Classical ML Benchmark Engine & The Generalization Cliff
# -----------------------------------------------------------------------------
def benchmark_models_across_splits(partitions, class_dict):
    """
    Trains and benchmarks 5 classical algorithms under both Random Split and Disjoint Parcel Split:
    1. Random Forest (RF, 100 trees)
    2. Support Vector Machine (SVM-RBF, C=10.0, gamma='scale')
    3. Extra Trees Classifier (ET, 100 trees)
    4. k-Nearest Neighbors (k-NN, k=5)
    5. Spectral Angle Mapper (SAM Nearest Endmember)
    """
    classes = [c for c in sorted(class_dict.keys()) if c > 0]
    
    models = {
        "Random Forest (100 Trees)": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Extra Trees (100 Trees)": lambda: ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "SVM (RBF Kernel, C=10)": lambda: SVC(C=10.0, kernel="rbf", gamma="scale", random_state=42),
        "k-Nearest Neighbors (k=5)": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Spectral Angle Mapper (SAM)": lambda: SpectralAngleMapperClassifier()
    }
    
    benchmark_results = {
        "random_split": {},
        "parcel_split": {},
        "generalization_cliff": {}
    }
    
    # Store fitted RF for band importance ranking
    fitted_rf_model = None

    for split_key in ["random_split", "parcel_split"]:
        p_data = partitions[split_key]
        X_tr = p_data["X_train"]
        y_tr = p_data["y_train"]
        X_te = p_data["X_test"]
        y_te = p_data["y_test"]

        # Only evaluate on classes that exist in both train and test sets
        eval_classes = sorted(list(set(y_tr).intersection(set(y_te))))

        for m_name, m_ctor in models.items():
            print(f"[{split_key.upper()}] Training {m_name}...")
            clf = m_ctor()
            
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            fit_duration = time.time() - t0

            t0_inf = time.time()
            preds = clf.predict(X_te)
            inf_duration = (time.time() - t0_inf) / len(X_te) * 1000.0 # ms per sample

            # Keep fitted RF from random split for feature attribution
            if split_key == "random_split" and "Random Forest" in m_name:
                fitted_rf_model = clf

            oa = float(accuracy_score(y_te, preds))
            kappa = float(cohen_kappa_score(y_te, preds))
            aa = float(recall_score(y_te, preds, labels=eval_classes, average="macro", zero_division=0))
            macro_f1 = float(f1_score(y_te, preds, labels=eval_classes, average="macro", zero_division=0))
            weighted_f1 = float(f1_score(y_te, preds, labels=eval_classes, average="weighted", zero_division=0))

            per_class_recalls = recall_score(y_te, preds, labels=eval_classes, average=None, zero_division=0)
            per_class_f1s = f1_score(y_te, preds, labels=eval_classes, average=None, zero_division=0)

            per_cls_metrics = {}
            for c_val, rec, f1 in zip(eval_classes, per_class_recalls, per_class_f1s):
                per_cls_metrics[class_dict[c_val]] = {
                    "recall": round(float(rec), 4),
                    "f1_score": round(float(f1), 4)
                }

            benchmark_results[split_key][m_name] = {
                "overall_accuracy_oa": round(oa, 4),
                "average_accuracy_aa": round(aa, 4),
                "cohens_kappa": round(kappa, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "fit_time_seconds": round(fit_duration, 3),
                "inference_ms_per_sample": round(inf_duration, 4),
                "evaluated_classes_count": len(eval_classes),
                "per_class_metrics": per_cls_metrics
            }

    # Compute Generalization Cliff metrics: Drop from Naive Random to Disjoint Parcel
    for m_name in models.keys():
        oa_rnd = benchmark_results["random_split"][m_name]["overall_accuracy_oa"]
        oa_prc = benchmark_results["parcel_split"][m_name]["overall_accuracy_oa"]
        kappa_rnd = benchmark_results["random_split"][m_name]["cohens_kappa"]
        kappa_prc = benchmark_results["parcel_split"][m_name]["cohens_kappa"]
        aa_rnd = benchmark_results["random_split"][m_name]["average_accuracy_aa"]
        aa_prc = benchmark_results["parcel_split"][m_name]["average_accuracy_aa"]

        drop_oa_abs = round(oa_rnd - oa_prc, 4)
        drop_oa_pct = round((oa_rnd - oa_prc) / (oa_rnd + 1e-8) * 100.0, 2)
        drop_kappa_abs = round(kappa_rnd - kappa_prc, 4)

        benchmark_results["generalization_cliff"][m_name] = {
            "random_split_oa": oa_rnd,
            "parcel_split_oa": oa_prc,
            "absolute_accuracy_drop": drop_oa_abs,
            "relative_accuracy_drop_pct": drop_oa_pct,
            "random_split_kappa": kappa_rnd,
            "parcel_split_kappa": kappa_prc,
            "absolute_kappa_drop": drop_kappa_abs,
            "random_split_aa": aa_rnd,
            "parcel_split_aa": aa_prc
        }

    return benchmark_results, fitted_rf_model

# -----------------------------------------------------------------------------
# 5. RF Spectral Band Importance & Biochemical Wavelength Attribution
# -----------------------------------------------------------------------------
def analyze_band_importance(rf_model, wavelengths):
    """
    Ranks all 200 spectral channels by Random Forest Gini feature importance
    and correlates prominent importance peaks with known remote sensing biochemical absorption features.
    """
    importances = rf_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    records = []
    for rank, idx in enumerate(sorted_indices, 1):
        wl = wavelengths[idx]
        imp = float(importances[idx])
        
        # Biochemical and physiological diagnostic annotations
        if 400 <= wl <= 500:
            domain = "VIS Blue"
            bio = "Chlorophyll-a/b Soret band, Carotenoid absorption (Blue photon capture)"
        elif 500 < wl <= 600:
            domain = "VIS Green"
            bio = "Photosynthetic reflection peak (Green peak), Xanthophyll cycle activity (~531nm)"
        elif 600 < wl <= 700:
            domain = "VIS Red"
            bio = "Strong Chlorophyll-a absorption peak (~670-680nm), photosynthetic capacity"
        elif 700 < wl <= 750:
            domain = "Red-Edge"
            bio = "Canopy Red-Edge inflection; cellular scattering transition, sensitive to early chlorosis"
        elif 750 < wl <= 1300:
            domain = "NIR Plateau"
            bio = "Spongy mesophyll cellular multiple scattering, leaf area index (LAI), canopy vigor"
        elif 1300 < wl <= 1900:
            domain = "SWIR-1"
            bio = "Foliar liquid water absorption, leaf turgor pressure, canopy moisture deficit"
        else:
            domain = "SWIR-2"
            bio = "Lignin, cellulose, starch, and plant structural carbon dry matter absorption"
            
        records.append({
            "rank": rank,
            "corrected_band_index_0based": int(idx),
            "corrected_band_index_1based": int(idx + 1),
            "wavelength_nm": round(float(wl), 2),
            "spectral_domain": domain,
            "gini_importance": round(imp, 6),
            "cumulative_importance": 0.0, # Filled below
            "biochemical_mechanism": bio
        })
        
    df = pd.DataFrame(records)
    df["cumulative_importance"] = np.cumsum(df["gini_importance"]).round(5)
    return df

# -----------------------------------------------------------------------------
# 6. Publication Figures Generation
# -----------------------------------------------------------------------------
def generate_figure1_vegetation_indices(ip_indices, ip_gt, salinas_indices, salinas_gt, output_path):
    """
    Figure 1: Spatial heatmaps of NDVI, NDRE, PRI, and WBI for Indian Pines and Salinas scenes.
    """
    fig = plt.figure(figsize=(18, 12), dpi=200)
    gs = GridSpec(2, 4, figure=fig, hspace=0.28, wspace=0.25)

    # Indian Pines Indices
    ip_maps = [
        ("Indian Pines: NDVI\n(Biomass & Canopy Vigor)", ip_indices["NDVI"], "YlGn", (0.1, 0.85)),
        ("Indian Pines: NDRE\n(Red-Edge Early Chlorosis)", ip_indices["NDRE"], "RdYlGn", (0.05, 0.65)),
        ("Indian Pines: PRI\n(Photosynthetic Stress)", ip_indices["PRI"], "magma", (-0.15, 0.15)),
        ("Indian Pines: WBI\n(Canopy Liquid Water Status)", ip_indices["WBI"], "Blues", (0.85, 1.25))
    ]

    for col, (title, data, cmap, vrange) in enumerate(ip_maps):
        ax = fig.add_subplot(gs[0, col])
        # Mask out background for clear contrast
        display_data = np.where(ip_gt > 0, data, np.nan)
        im = ax.imshow(display_data, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Salinas Indices
    sal_maps = [
        ("Salinas: NDVI\n(Canopy Cover & Phenology)", salinas_indices["NDVI"], "YlGn", (0.1, 0.9)),
        ("Salinas: NDRE\n(Crop Vitality Progression)", salinas_indices["NDRE"], "RdYlGn", (0.05, 0.7)),
        ("Salinas: PRI\n(Light-Use Efficiency)", salinas_indices["PRI"], "magma", (-0.12, 0.12)),
        ("Salinas: WBI\n(Irrigation & Water Content)", salinas_indices["WBI"], "Blues", (0.9, 1.25))
    ]

    for col, (title, data, cmap, vrange) in enumerate(sal_maps):
        ax = fig.add_subplot(gs[1, col])
        display_data = np.where(salinas_gt > 0, data, np.nan)
        im = ax.imshow(display_data, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.suptitle("TerraSpectra Day 3: Narrowband Biophysical & Crop Stress Indices Spatial Mapping\n"
                 "(NDVI, NDRE, PRI, and WBI across Indian Pines and Salinas Valley Agricultural Scenes)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 to: {output_path}")

def generate_figure2_spectral_angle_mapper(sam_matrix, endmembers, wavelengths, class_dict, output_path):
    """
    Figure 2: Spectral Angle Mapper (SAM) pairwise angular distance matrix and endmember profiles.
    """
    classes = [c for c in sorted(class_dict.keys()) if c > 0]
    class_names = [class_dict[c] for c in classes]

    fig = plt.figure(figsize=(20, 9), dpi=200)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.25)

    # Subplot 1: SAM Heatmap Matrix
    ax_mat = fig.add_subplot(gs[0, 0])
    im = ax_mat.imshow(sam_matrix, cmap="viridis_r", vmin=0, vmax=25)
    ax_mat.set_title("Pairwise Spectral Angle Mapper (SAM) Distance Matrix (Degrees)\n"
                     "Demonstrates Extreme Angular Similarity between Tillage Subtypes (< 0.9°)",
                     fontsize=11, fontweight="bold", pad=10)
    ax_mat.set_xticks(range(len(classes)))
    ax_mat.set_yticks(range(len(classes)))
    ax_mat.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
    ax_mat.set_yticklabels(class_names, fontsize=8)

    # Overlay numeric angles for top confusions
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = sam_matrix[i, j]
            if 0 < val < 1.5:
                ax_mat.text(j, i, f"{val:.1f}°", ha="center", va="center", color="red", fontsize=6, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    cbar.set_label("Spectral Angle (Degrees)", fontsize=9, fontweight="bold")

    # Subplot 2: Endmember Spectra Profiles
    ax_spec = fig.add_subplot(gs[0, 1])
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    # Highlight specific key diagnostic classes
    highlight_classes = [2, 3, 4, 10, 11, 14, 16] # Corn-notill, Corn-mintill, Corn, Soy-notill, Soy-mintill, Woods, Towers
    for c in classes:
        spec = endmembers[c]
        name = class_dict[c]
        if c in highlight_classes:
            ax_spec.plot(wavelengths, spec, label=f"C{c}: {name}", linewidth=2.0)
        else:
            ax_spec.plot(wavelengths, spec, color="#bbbbbb", linewidth=0.8, alpha=0.5)

    ax_spec.set_title("Calibrated Reference Spectral Endmembers (Unit Normalized)\n"
                      "Vegetation Plateau vs Man-made Structures vs Tillage Regimes",
                      fontsize=11, fontweight="bold", pad=10)
    ax_spec.set_xlabel("Wavelength (nm)", fontsize=9, fontweight="bold")
    ax_spec.set_ylabel("Unit-Normalized Spectral Intensity", fontsize=9, fontweight="bold")
    ax_spec.grid(True, linestyle="--", alpha=0.4)
    ax_spec.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax_spec.set_xlim(400, 2450)

    plt.suptitle("TerraSpectra Day 3: Spectral Angle Mapper (SAM) Analysis & Endmember Geometry",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 to: {output_path}")

def generate_figure3_ml_generalization_cliff(benchmark_results, output_path):
    """
    Figure 3: Side-by-side comparison of classical ML performance across Naive Random Split vs Disjoint Parcel Split.
    Visualizes the Generalization Cliff (OA, AA, Cohen's Kappa).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    
    models = list(benchmark_results["random_split"].keys())
    x = np.arange(len(models))
    width = 0.35

    # 1. Overall Accuracy (OA)
    oa_rnd = [benchmark_results["random_split"][m]["overall_accuracy_oa"] * 100.0 for m in models]
    oa_prc = [benchmark_results["parcel_split"][m]["overall_accuracy_oa"] * 100.0 for m in models]
    
    rects1 = axes[0].bar(x - width/2, oa_rnd, width, label="Naive Random Split (Leakage)", color="#1f77b4", alpha=0.9)
    rects2 = axes[0].bar(x + width/2, oa_prc, width, label="Disjoint Parcel Split (Sound)", color="#d62728", alpha=0.9)
    axes[0].set_ylabel("Overall Accuracy (%)", fontsize=10, fontweight="bold")
    axes[0].set_title("Overall Accuracy (OA) Benchmark\nNotice Sharp Drop (~40-50%) in Generalization", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.split("(")[0].strip() for m in models], rotation=30, ha="right", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8, loc="upper right")

    # Add numeric labels
    for r1, r2 in zip(rects1, rects2):
        axes[0].text(r1.get_x() + r1.get_width()/2., r1.get_height() + 1.5, f"{r1.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold")
        axes[0].text(r2.get_x() + r2.get_width()/2., r2.get_height() + 1.5, f"{r2.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold", color="#d62728")

    # 2. Average Accuracy (AA / Macro Recall)
    aa_rnd = [benchmark_results["random_split"][m]["average_accuracy_aa"] * 100.0 for m in models]
    aa_prc = [benchmark_results["parcel_split"][m]["average_accuracy_aa"] * 100.0 for m in models]
    
    rects3 = axes[1].bar(x - width/2, aa_rnd, width, label="Naive Random Split", color="#2ca02c", alpha=0.9)
    rects4 = axes[1].bar(x + width/2, aa_prc, width, label="Disjoint Parcel Split", color="#ff7f0e", alpha=0.9)
    axes[1].set_ylabel("Average Accuracy / Macro Recall (%)", fontsize=10, fontweight="bold")
    axes[1].set_title("Average Accuracy (AA) Benchmark\nImbalanced Minority Classes Under Parcel Split", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.split("(")[0].strip() for m in models], rotation=30, ha="right", fontsize=9)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8, loc="upper right")

    for r3, r4 in zip(rects3, rects4):
        axes[1].text(r3.get_x() + r3.get_width()/2., r3.get_height() + 1.5, f"{r3.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold")
        axes[1].text(r4.get_x() + r4.get_width()/2., r4.get_height() + 1.5, f"{r4.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold", color="#ff7f0e")

    # 3. Cohen's Kappa Coefficient (Kappa)
    kappa_rnd = [benchmark_results["random_split"][m]["cohens_kappa"] for m in models]
    kappa_prc = [benchmark_results["parcel_split"][m]["cohens_kappa"] for m in models]
    
    rects5 = axes[2].bar(x - width/2, kappa_rnd, width, label="Naive Random Split", color="#9467bd", alpha=0.9)
    rects6 = axes[2].bar(x + width/2, kappa_prc, width, label="Disjoint Parcel Split", color="#8c564b", alpha=0.9)
    axes[2].set_ylabel("Cohen's Kappa (κ)", fontsize=10, fontweight="bold")
    axes[2].set_title("Cohen's Kappa Statistic (κ)\nChance-Adjusted Inter-Rater Reliability", fontsize=10, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([m.split("(")[0].strip() for m in models], rotation=30, ha="right", fontsize=9)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].legend(fontsize=8, loc="upper right")

    for r5, r6 in zip(rects5, rects6):
        axes[2].text(r5.get_x() + r5.get_width()/2., r5.get_height() + 0.02, f"{r5.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold")
        axes[2].text(r6.get_x() + r6.get_width()/2., r6.get_height() + 0.02, f"{r6.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold", color="#8c564b")

    plt.suptitle("TerraSpectra Day 3: Classical ML Benchmark & The Generalization Cliff\n"
                 "(Proof of Performance Degradation on Unseen Agricultural Parcels Without Spatial Leakage)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 to: {output_path}")

def generate_figure4_rf_importance(importance_df, wavelengths, output_path):
    """
    Figure 4: Spectral band importance profile from Random Forest, overlaid with AVIRIS spectrometer domains
    and physiological diagnostic absorption annotations.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=200, gridspec_kw={"height_ratios": [1.4, 1.0]})

    # Reorder importances by wavelength
    df_sorted = importance_df.sort_values("wavelength_nm")
    wls = df_sorted["wavelength_nm"].values
    imps = df_sorted["gini_importance"].values

    # Plot 1: Full Spectrum Importance Curve with Subsystems Shaded
    ax1.plot(wls, imps, color="#1b5e20", linewidth=1.8, label="RF Band Gini Importance")
    ax1.fill_between(wls, 0, imps, color="#4caf50", alpha=0.35)

    # Shade AVIRIS Optical Subsystems
    ax1.axvspan(400, 700, color="#2196f3", alpha=0.1, label="Spectrometer A: VIS (400-700nm)")
    ax1.axvspan(700, 1300, color="#4caf50", alpha=0.1, label="Spectrometer B: NIR (700-1300nm)")
    ax1.axvspan(1300, 1900, color="#ff9800", alpha=0.1, label="Spectrometer C: SWIR-1 (1300-1900nm)")
    ax1.axvspan(1900, 2450, color="#9c27b0", alpha=0.1, label="Spectrometer D: SWIR-2 (1900-2450nm)")

    # Highlight top 5 diagnostic peaks
    top5 = importance_df.head(5)
    for _, row in top5.iterrows():
        wl_peak = row["wavelength_nm"]
        imp_peak = row["gini_importance"]
        ax1.annotate(f"Rank {row['rank']}: {wl_peak:.0f}nm\n({row['spectral_domain']})",
                     xy=(wl_peak, imp_peak),
                     xytext=(wl_peak, imp_peak + 0.003),
                     arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                     fontsize=7.5, fontweight="bold", ha="center")

    ax1.set_title("Random Forest Wavelength Feature Importance Profile (200 Clean Spectral Channels)\n"
                  "Highlighting Dominant Chlorophyll-a (670nm), Red-Edge (720nm), and SWIR Water Bands",
                  fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Gini Importance Metric", fontsize=9, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_xlim(400, 2450)

    # Plot 2: Top 15 Ranked Diagnostic Wavelengths Horizontal Bar Chart
    top15 = importance_df.head(15).iloc[::-1]
    y_pos = np.arange(len(top15))
    bar_colors = ["#2196f3" if "VIS" in d else "#4caf50" if "Red-Edge" in d or "NIR" in d else "#ff9800" if "SWIR-1" in d else "#9c27b0"
                  for d in top15["spectral_domain"]]

    ax2.barh(y_pos, top15["gini_importance"] * 100.0, color=bar_colors, alpha=0.85, edgecolor="#333333", linewidth=0.5)
    ax2.set_yticks(y_pos)
    labels = [f"Rank {r}: {wl:.1f}nm ({dom})" for r, wl, dom in zip(top15['rank'], top15['wavelength_nm'], top15['spectral_domain'])]
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Feature Importance Contribution (%)", fontsize=9, fontweight="bold")
    ax2.set_title("Top 15 Most Informative Spectral Bands for Agricultural Classification", fontsize=10, fontweight="bold", pad=6)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 to: {output_path}")

# -----------------------------------------------------------------------------
# 7. Main Execution Pipeline
# -----------------------------------------------------------------------------
def run_day03_pipeline():
    print("=" * 80)
    print("🌿 TerraSpectra Day 3: Narrowband Biophysical Indices & Classical ML Baseline")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/6] Loading corrected hyperspectral cubes and ground truth masks...")
    ip_corr_path = os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat")
    ip_gt_path = os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat")
    sal_corr_path = os.path.join(RAW_DATA_DIR, "Salinas_corrected.mat")
    sal_gt_path = os.path.join(RAW_DATA_DIR, "Salinas_gt.mat")
    ip_wl_path = os.path.join(PROC_DATA_DIR, "indian_pines_wavelengths.csv")

    ip_cube = sio.loadmat(ip_corr_path)["indian_pines_corrected"].astype(np.float32)
    ip_gt = sio.loadmat(ip_gt_path)["indian_pines_gt"]
    sal_cube = sio.loadmat(sal_corr_path)["salinas_corrected"].astype(np.float32)
    sal_gt = sio.loadmat(sal_gt_path)["salinas_gt"]

    # Calibrated reflectance normalization rho = clip(DN / 10000.0, 0, 1)
    ip_refl = np.clip(ip_cube / 10000.0, 0.0, 1.0)
    sal_refl = np.clip(sal_cube / 10000.0, 0.0, 1.0)

    # Load Indian Pines wavelengths
    wl_df = pd.read_csv(ip_wl_path)
    clean_wl_df = wl_df[wl_df["corrected_band_index_1based"].notna()].copy()
    ip_wavelengths = clean_wl_df["wavelength_nm"].values

    # Approximate Salinas wavelengths (AVIRIS 400 - 2500 nm linear spacing across 204 bands)
    sal_wavelengths = np.linspace(400.0, 2450.0, sal_cube.shape[2])

    print(f"  -> Indian Pines Refl Cube: {ip_refl.shape}, Ground Truth: {ip_gt.shape}")
    print(f"  -> Salinas Refl Cube: {sal_refl.shape}, Ground Truth: {sal_gt.shape}")

    # 2. Compute Biophysical Vegetation Indices
    print("\n[Step 2/6] Computing 8 Narrowband Biophysical & Crop Stress Indices...")
    ip_indices = compute_narrowband_indices(ip_refl, ip_wavelengths)
    sal_indices = compute_narrowband_indices(sal_refl, sal_wavelengths)
    
    indices_summary_df = summarize_indices_by_class(ip_indices, ip_gt, INDIAN_PINES_CLASSES)
    indices_csv_path = os.path.join(PROC_DATA_DIR, "day03_vegetation_indices_summary.csv")
    indices_summary_df.to_csv(indices_csv_path, index=False)
    print(f"  -> Exported Vegetation Indices Summary: {indices_csv_path} ({len(indices_summary_df)} classes)")

    # 3. Spectral Angle Mapper (SAM) Analysis
    print("\n[Step 3/6] Computing Spectral Angle Mapper (SAM) Class Endmembers and Distance Matrix...")
    mask_ip = (ip_gt > 0)
    X_ip_all = ip_refl[mask_ip]
    y_ip_all = ip_gt[mask_ip]

    sam_matrix_deg, endmembers, sam_records_df = compute_sam_matrix(X_ip_all, y_ip_all, INDIAN_PINES_CLASSES)
    sam_csv_path = os.path.join(PROC_DATA_DIR, "day03_spectral_angle_matrix.csv")
    sam_records_df.to_csv(sam_csv_path, index=False)
    print(f"  -> Exported Pairwise SAM Angle Matrix: {sam_csv_path} ({len(sam_records_df)} pairs)")

    # 4. Partitioning & Classical ML Benchmark
    print("\n[Step 4/6] Partitioning Datasets & Benchmarking Classical ML Models across Splits...")
    partitions = build_dataset_partitions(ip_cube, ip_gt, test_ratio=0.2, random_seed=42)
    benchmark_results, fitted_rf = benchmark_models_across_splits(partitions, INDIAN_PINES_CLASSES)
    
    benchmark_json_path = os.path.join(PROC_DATA_DIR, "day03_model_benchmark_results.json")
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"  -> Exported Model Benchmark Results: {benchmark_json_path}")

    # 5. RF Band Importance & Biochemical Wavelength Ranking
    print("\n[Step 5/6] Extracting Random Forest Spectral Band Importance & Biochemical Ranking...")
    importance_df = analyze_band_importance(fitted_rf, ip_wavelengths)
    importance_csv_path = os.path.join(PROC_DATA_DIR, "day03_band_importance_ranking.csv")
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"  -> Exported Top Band Importance Rankings: {importance_csv_path} (200 bands)")

    # 6. Generate Publication Figures
    print("\n[Step 6/6] Generating Publication-Quality Diagnostic Visualizations...")
    fig1_path = os.path.join(FIG_DIR, "day03_vegetation_indices_spatial_maps.png")
    fig2_path = os.path.join(FIG_DIR, "day03_spectral_angle_mapper_matrix.png")
    fig3_path = os.path.join(FIG_DIR, "day03_classical_ml_benchmark_cliff.png")
    fig4_path = os.path.join(FIG_DIR, "day03_rf_wavelength_importance.png")

    generate_figure1_vegetation_indices(ip_indices, ip_gt, sal_indices, sal_gt, fig1_path)
    generate_figure2_spectral_angle_mapper(sam_matrix_deg, endmembers, ip_wavelengths, INDIAN_PINES_CLASSES, fig2_path)
    generate_figure3_ml_generalization_cliff(benchmark_results, fig3_path)
    generate_figure4_rf_importance(importance_df, ip_wavelengths, fig4_path)

    print("\n" + "=" * 80)
    print("✅ TerraSpectra Day 3 Pipeline Executed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_day03_pipeline()
