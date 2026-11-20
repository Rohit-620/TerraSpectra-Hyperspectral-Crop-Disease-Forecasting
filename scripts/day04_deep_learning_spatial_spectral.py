"""
TerraSpectra: Day 4 - Spatial-Spectral Deep Learning Architectures (1D-CNN vs 2D-CNN vs 3D-CNN & HybridSN)
Benchmarking Spatial Autocorrelation, Receptive Fields & Overcoming the Generalization Cliff
Author: Rohit (TerraSpectra ML Team)
Working Branch: Rohit-EDA-Analysis
Milestone: Day 4 - Deep Learning Spatial-Spectral Architecture Suite & Generalization Benchmark
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
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
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
# Indian Pines Agricultural Taxonomy & Visualization Palette
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

# Distinct 17-color categorical palette (index 0 is dark background)
IP_HEX_COLORS = [
    "#111111",  # 0: Background
    "#e6194B",  # 1: Alfalfa (Red)
    "#3cb44b",  # 2: Corn-notill (Green)
    "#ffe119",  # 3: Corn-mintill (Yellow)
    "#4363d8",  # 4: Corn (Blue)
    "#f58231",  # 5: Grass-pasture (Orange)
    "#911eb4",  # 6: Grass-trees (Purple)
    "#42d4f4",  # 7: Grass-pasture-mowed (Cyan)
    "#f032e6",  # 8: Hay-windrowed (Magenta)
    "#bfef45",  # 9: Oats (Lime)
    "#fabed4",  # 10: Soybean-notill (Pink)
    "#469990",  # 11: Soybean-mintill (Teal)
    "#dcbeff",  # 12: Soybean-clean (Lavender)
    "#9A6324",  # 13: Wheat (Brown)
    "#fffac8",  # 14: Woods (Beige)
    "#800000",  # 15: Buildings-Grass-Trees-Drives (Maroon)
    "#aaffc3"   # 16: Stone-Steel-Towers (Mint)
]
IP_CMAP = ListedColormap(IP_HEX_COLORS)

# -----------------------------------------------------------------------------
# Deep Learning Spatial-Spectral Architecture Suite
# -----------------------------------------------------------------------------
class Spectral1DCNN(nn.Module):
    """
    1D-Spectral CNN Baseline (Pixel-Wise Spectral Architecture)
    Input shape: (batch_size, 1, 200) or (batch_size, 200)
    Learns continuous 1D absorption depth, slope, and spectral inflections.
    Possesses zero spatial awareness (1x1 spatial receptive field).
    """
    def __init__(self, in_bands=200, num_classes=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class Spatial2DCNN(nn.Module):
    """
    2D-Spatial CNN Baseline (Dimension-Reduced Spatial Context Architecture)
    Input shape: (batch_size, C=30, k, k) or (batch_size, k, k, C=30)
    Learns spatial texture, parcel boundaries, and field homogeneity.
    Treats spectral channels as flat feature maps, collapsing inter-band continuum.
    """
    def __init__(self, in_channels=30, num_classes=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 4 and x.shape[3] == 30 and x.shape[1] != 30:
            x = x.permute(0, 3, 1, 2)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class SpatialSpectral3DCNN(nn.Module):
    """
    3D-Spatial-Spectral Volumetric CNN
    Input shape: (batch_size, 1, D=30, k=9, k=9) or (batch_size, k=9, k=9, D=30)
    Simultaneously convolves across spatial coordinates and continuous spectral depth.
    """
    def __init__(self, in_channels=1, num_classes=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        pooled = self.pool(f3)
        flat = pooled.view(pooled.size(0), -1)
        return self.classifier(flat)


class HybridSN(nn.Module):
    """
    HybridSN: Hybrid 3D-2D Spatial-Spectral Convolutional Network (Roy et al., IEEE GRSL 2019)
    Input shape: (batch_size, 1, D=30, k=9, k=9) or (batch_size, k=9, k=9, D=30)
    Applies stacked 3D convolutions to preserve low-level spatial-spectral joint relationships,
    then reshapes spectral depth into 2D channels to extract rich spatial abstractions.
    """
    def __init__(self, in_channels=1, num_classes=16):
        super().__init__()
        # 3D Convolutional Hierarchy
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        # 2D Convolutional Spatial Stage
        # Spatial-spectral tensor evolution for (1, 30, 9, 9):
        # 1. Conv3D (7,3,3) -> (8, 24, 7, 7)
        # 2. Conv3D (5,3,3) -> (16, 20, 5, 5)
        # 3. Conv3D (3,3,3) -> (32, 18, 3, 3)
        # Reshaped to 2D planar map -> (32 * 18 = 576, 3, 3)
        self.conv2d = nn.Sequential(
            nn.Conv2d(32 * 18, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        f1 = self.conv3d_1(x)
        f2 = self.conv3d_2(f1)
        f3 = self.conv3d_3(f2)
        b, c, d, h, w = f3.shape
        f3_2d = f3.view(b, c * d, h, w)
        f4 = self.conv2d(f3_2d)
        flat = f4.view(b, -1)
        return self.classifier(flat)


# -----------------------------------------------------------------------------
# Data Ingestion, Dimensionality Reduction & Patch Generation
# -----------------------------------------------------------------------------
def load_and_preprocess_hyperspectral_data():
    """
    Loads calibrated Indian Pines dataset, performs PCA reduction to 30 components,
    and extracts spatial-spectral patches and flat spectral vectors.
    """
    print("[1/6] Loading calibrated Indian Pines dataset and fitting PCA-30 projection...")
    mat_cube = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat"))
    mat_gt = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat"))

    cube = [mat_cube[k] for k in mat_cube if not k.startswith("__")][0]
    gt = [mat_gt[k] for k in mat_gt if not k.startswith("__")][0]

    H, W, B = cube.shape
    normalized_cube = cube.astype(np.float32) / 10000.0

    # Fit PCA across all spatial pixels
    flat_pixels = normalized_cube.reshape(-1, B)
    pca = PCA(n_components=30, random_state=42)
    flat_pca = pca.fit_transform(flat_pixels)
    cube_pca = flat_pca.reshape(H, W, 30)
    explained_var = float(np.sum(pca.explained_variance_ratio_))

    print(f"  -> Cube dimensions: {H}x{W} with {B} calibrated channels.")
    print(f"  -> PCA-30 reduction explains {explained_var * 100.0:.3f}% total spectral variance.")

    # Extract labeled coordinates
    labeled_mask = (gt > 0)
    coords = np.argwhere(labeled_mask)
    labels = gt[labeled_mask] - 1  # Convert 1-indexed to 0-indexed [0..15]
    n_samples = len(labels)

    # 1D Spectral Vectors (full 200 bands)
    X_1d = normalized_cube[labeled_mask]

    # Spatial-spectral patches for patch_size = 9
    patch_size = 9
    m = patch_size // 2
    padded_pca = np.pad(cube_pca, ((m, m), (m, m), (0, 0)), mode="reflect")
    X_patches_9 = np.zeros((n_samples, patch_size, patch_size, 30), dtype=np.float32)

    for i, (r, c) in enumerate(coords):
        X_patches_9[i] = padded_pca[r:r + patch_size, c:c + patch_size, :]

    print(f"  -> Extracted {n_samples} spatial patches of size {patch_size}x{patch_size}x30.")
    return {
        "cube_raw": cube,
        "cube_normalized": normalized_cube,
        "cube_pca": cube_pca,
        "gt": gt,
        "coords": coords,
        "labels": labels,
        "X_1d": X_1d,
        "X_patches_9": X_patches_9,
        "explained_var_ratio": explained_var,
        "shape": (H, W, B)
    }


def construct_evaluation_splits(gt, coords, labels, random_seed=42):
    """
    Constructs both:
    1. Naive Random Split (80% train / 20% test stratified)
    2. Spatial Disjoint Parcel Split (Intact connected-component parcels)
    """
    print("[2/6] Partitioning experimental evaluation splits (Random vs Disjoint Parcel)...")
    H, W = gt.shape
    n_samples = len(labels)

    # 1. Naive Random Split
    idx_rnd_train, idx_rnd_test = train_test_split(
        np.arange(n_samples), test_size=0.20, random_state=random_seed, stratify=labels
    )

    # 2. Spatial Disjoint Parcel Split via Connected Components
    labeled_mask = (gt > 0)
    parcel_labels, num_parcels = ndi.label(labeled_mask)
    parcel_records = []
    for p_id in range(1, num_parcels + 1):
        p_mask = (parcel_labels == p_id)
        p_gt_vals = gt[p_mask] - 1
        dom_cls = int(np.bincount(p_gt_vals).argmax())
        parcel_records.append((p_id, dom_cls, int(np.sum(p_mask))))

    np.random.seed(random_seed)
    test_parcels = []
    for cls_id in np.unique(labels):
        cls_parcels = [p[0] for p in parcel_records if p[1] == cls_id]
        if len(cls_parcels) > 1:
            n_test = max(1, int(len(cls_parcels) * 0.20))
            test_parcels.extend(list(np.random.choice(cls_parcels, n_test, replace=False)))

    test_parcel_set = set(test_parcels)
    parcel_test_mask = np.zeros((H, W), dtype=bool)
    for p_id in test_parcel_set:
        parcel_test_mask[parcel_labels == p_id] = True

    idx_prc_test = np.array([i for i, (r, c) in enumerate(coords) if parcel_test_mask[r, c]])
    idx_prc_train = np.array([i for i, (r, c) in enumerate(coords) if not parcel_test_mask[r, c]])

    print(f"  -> Regime A (Random Split):   {len(idx_rnd_train)} train / {len(idx_rnd_test)} test samples.")
    print(f"  -> Regime B (Parcel Split):   {len(idx_prc_train)} train / {len(idx_prc_test)} test samples.")

    return {
        "random": {"train": idx_rnd_train, "test": idx_rnd_test},
        "parcel": {"train": idx_prc_train, "test": idx_prc_test},
        "num_parcels": num_parcels,
        "test_parcel_count": len(test_parcel_set)
    }


# -----------------------------------------------------------------------------
# Deep Learning Training & Benchmarking Engine
# -----------------------------------------------------------------------------
def train_and_evaluate_model(model_class, model_name, X_data, y_data, splits, regime_name,
                             epochs=10, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=42):
    """
    Instantiates, trains, and evaluates a deep neural network on a specified split regime.
    Tracks epoch histories, latency, and comprehensive multi-class metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx = splits[regime_name]["train"]
    test_idx = splits[regime_name]["test"]

    X_tr, y_tr = torch.from_numpy(X_data[train_idx]), torch.from_numpy(y_data[train_idx])
    X_te, y_te = torch.from_numpy(X_data[test_idx]), torch.from_numpy(y_data[test_idx])

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    model = model_class()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": []
    }

    start_fit_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_tr = 0
        total_tr = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = out.argmax(dim=1)
            correct_tr += (preds == by).sum().item()
            total_tr += len(by)

        scheduler.step()
        ep_tr_loss = running_loss / total_tr
        ep_tr_acc = correct_tr / total_tr

        # Validation evaluation
        model.eval()
        running_te_loss = 0.0
        correct_te = 0
        total_te = 0
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx)
                loss = criterion(out, by)
                running_te_loss += loss.item() * len(by)
                preds = out.argmax(dim=1)
                correct_te += (preds == by).sum().item()
                total_te += len(by)

        ep_te_loss = running_te_loss / total_te
        ep_te_acc = correct_te / total_te
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(ep)
        history["train_loss"].append(round(ep_tr_loss, 4))
        history["train_acc"].append(round(ep_tr_acc, 4))
        history["test_loss"].append(round(ep_te_loss, 4))
        history["test_acc"].append(round(ep_te_acc, 4))
        history["lr"].append(current_lr)

    fit_duration = time.time() - start_fit_time

    # Inference Latency Benchmark
    model.eval()
    all_preds = []
    latencies = []
    with torch.no_grad():
        for bx, by in test_loader:
            t0 = time.perf_counter()
            out = model(bx)
            preds = out.argmax(dim=1).cpu().numpy()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / len(by))
            all_preds.extend(preds)

    mean_inf_latency_ms = float(np.mean(latencies) * 1000.0)
    y_true = y_data[test_idx]
    y_pred = np.array(all_preds)

    # Core Agricultural Remote Sensing Metrics
    oa = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    recalls_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    aa = float(np.mean(recalls_per_class))

    # Class-wise breakdowns
    precisions_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1s_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    present_classes = np.unique(y_true)

    per_class_dict = {}
    for c_idx in range(16):
        c_name = INDIAN_PINES_CLASSES[c_idx + 1]
        if c_idx in present_classes:
            pos = int(np.where(present_classes == c_idx)[0][0])
            per_class_dict[c_name] = {
                "precision": round(float(precisions_per_class[pos]), 4),
                "recall": round(float(recalls_per_class[pos]), 4),
                "f1_score": round(float(f1s_per_class[pos]), 4)
            }
        else:
            per_class_dict[c_name] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    print(f"  [{model_name}] {regime_name.upper()} => OA: {oa * 100.0:.2f}% | AA: {aa * 100.0:.2f}% | "
          f"Kappa: {kappa:.4f} | Fit: {fit_duration:.1f}s | Latency: {mean_inf_latency_ms:.3f}ms")

    return {
        "model_name": model_name,
        "regime": regime_name,
        "overall_accuracy_oa": round(oa, 4),
        "average_accuracy_aa": round(aa, 4),
        "cohens_kappa": round(kappa, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "fit_time_seconds": round(fit_duration, 2),
        "inference_ms_per_sample": round(mean_inf_latency_ms, 4),
        "parameter_count": param_count,
        "per_class_metrics": per_class_dict,
        "history": history,
        "trained_model": model
    }


def execute_deep_learning_benchmark_suite(data_dict, splits):
    """
    Executes full benchmark suite across all 4 architectures and both regimes.
    """
    print("[3/6] Running Deep Learning Benchmark Suite across Random & Parcel Splitting Regimes...")
    architectures = [
        ("Spectral 1D-CNN", lambda: Spectral1DCNN(200, 16), data_dict["X_1d"]),
        ("Spatial 2D-CNN", lambda: Spatial2DCNN(30, 16), data_dict["X_patches_9"]),
        ("Spatial-Spectral 3D-CNN", lambda: SpatialSpectral3DCNN(1, 16), data_dict["X_patches_9"]),
        ("HybridSN (3D-2D CNN)", lambda: HybridSN(1, 16), data_dict["X_patches_9"])
    ]

    benchmark_results = {
        "random_split": {},
        "parcel_split": {},
        "generalization_cliff": {}
    }
    training_histories = {
        "random_split": {},
        "parcel_split": {}
    }
    trained_models = {}

    for name, m_fn, x_data in architectures:
        # Regime A: Random Split (8 epochs)
        res_rnd = train_and_evaluate_model(
            m_fn, name, x_data, data_dict["labels"], splits, "random", epochs=8
        )
        benchmark_results["random_split"][name] = res_rnd
        training_histories["random_split"][name] = res_rnd["history"]

        # Regime B: Disjoint Parcel Split (8 epochs)
        res_prc = train_and_evaluate_model(
            m_fn, name, x_data, data_dict["labels"], splits, "parcel", epochs=8
        )
        benchmark_results["parcel_split"][name] = res_prc
        training_histories["parcel_split"][name] = res_prc["history"]
        trained_models[name] = res_prc["trained_model"]

        # Compute Generalization Cliff Metrics
        oa_rnd = res_rnd["overall_accuracy_oa"]
        oa_prc = res_prc["overall_accuracy_oa"]
        abs_drop = round(oa_rnd - oa_prc, 4)
        rel_drop_pct = round((abs_drop / oa_rnd) * 100.0, 2)
        kappa_rnd = res_rnd["cohens_kappa"]
        kappa_prc = res_prc["cohens_kappa"]
        kappa_drop = round(kappa_rnd - kappa_prc, 4)

        benchmark_results["generalization_cliff"][name] = {
            "random_split_oa": oa_rnd,
            "parcel_split_oa": oa_prc,
            "absolute_accuracy_drop": abs_drop,
            "relative_accuracy_drop_pct": rel_drop_pct,
            "random_split_kappa": kappa_rnd,
            "parcel_split_kappa": kappa_prc,
            "absolute_kappa_drop": kappa_drop,
            "random_split_aa": res_rnd["average_accuracy_aa"],
            "parcel_split_aa": res_prc["average_accuracy_aa"],
            "parameter_count": res_rnd["parameter_count"]
        }

    return benchmark_results, training_histories, trained_models


# -----------------------------------------------------------------------------
# Spatial Patch Size Receptive Field Ablation
# -----------------------------------------------------------------------------
def run_patch_size_ablation(data_dict, splits):
    """
    Evaluates HybridSN across spatial patch sizes: 5x5, 9x9, and 13x13.
    Demonstrates spatial receptive field impact on parcel generalization.
    """
    print("[4/6] Executing Spatial Receptive Field Ablation (k in {5, 9, 13})...")
    cube_pca = data_dict["cube_pca"]
    gt = data_dict["gt"]
    coords = data_dict["coords"]
    labels = data_dict["labels"]
    n_samples = len(labels)

    patch_sizes = [5, 9, 13]
    ablation_records = []

    for k in patch_sizes:
        m = k // 2
        padded = np.pad(cube_pca, ((m, m), (m, m), (0, 0)), mode="reflect")
        X_k = np.zeros((n_samples, k, k, 30), dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            X_k[i] = padded[r:r + k, c:c + k, :]

        # Adapt model for patch size k
        # To maintain identical architecture across k, use Spatial2DCNN with adaptive pooling
        res = train_and_evaluate_model(
            lambda: Spatial2DCNN(30, 16),
            f"Spatial-2D-CNN ({k}x{k})",
            X_k, labels, splits, "parcel", epochs=6
        )

        ablation_records.append({
            "patch_window": f"{k}x{k}",
            "spatial_k": k,
            "receptive_field_pixels": k * k,
            "overall_accuracy_oa": res["overall_accuracy_oa"],
            "cohens_kappa": res["cohens_kappa"],
            "average_accuracy_aa": res["average_accuracy_aa"],
            "macro_f1": res["macro_f1"],
            "fit_time_seconds": res["fit_time_seconds"]
        })

    ablation_df = pd.DataFrame(ablation_records)
    ablation_path = os.path.join(PROC_DATA_DIR, "day04_patch_size_ablation.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print(f"  -> Saved patch size ablation results to {ablation_path}")
    return ablation_df


# -----------------------------------------------------------------------------
# Full Scene Spatial Thematic Inference
# -----------------------------------------------------------------------------
def generate_full_scene_prediction_maps(data_dict, trained_models):
    """
    Performs full-scene spatial raster prediction (145x145) across all labeled pixels.
    Reconstructs spatial prediction maps for 1D-CNN, 2D-CNN, 3D-CNN, and HybridSN.
    """
    print("[5/6] Performing full scene thematic spatial inference for all models...")
    H, W, B = data_dict["shape"]
    gt = data_dict["gt"]
    coords = data_dict["coords"]
    X_1d = data_dict["X_1d"]
    X_patches = data_dict["X_patches_9"]

    pred_maps = {
        "Ground Truth": gt.copy()
    }

    dl_input_map = {
        "Spectral 1D-CNN": (X_1d, 256),
        "Spatial 2D-CNN": (X_patches, 256),
        "Spatial-Spectral 3D-CNN": (X_patches, 256),
        "HybridSN (3D-2D CNN)": (X_patches, 256)
    }

    for name, model in trained_models.items():
        model.eval()
        x_in, b_size = dl_input_map[name]
        loader = DataLoader(TensorDataset(torch.from_numpy(x_in)), batch_size=b_size, shuffle=False)
        preds_all = []
        with torch.no_grad():
            for (bx,) in loader:
                p = model(bx).argmax(dim=1).cpu().numpy()
                preds_all.extend(p)

        scene_map = np.zeros((H, W), dtype=int)
        for idx, (r, c) in enumerate(coords):
            scene_map[r, c] = preds_all[idx] + 1  # Revert to 1-indexed classes
        pred_maps[name] = scene_map

    return pred_maps


# -----------------------------------------------------------------------------
# Publication Figures Generation Engine
# -----------------------------------------------------------------------------
def generate_publication_figures(data_dict, histories, benchmark_results, pred_maps, ablation_df):
    """
    Produces 4 high-resolution (300 DPI) publication-grade figures:
    1. day04_dl_training_curves.png
    2. day04_deep_learning_benchmark_cliff.png
    3. day04_full_scene_thematic_maps.png
    4. day04_patch_size_ablation_and_confusion.png
    """
    print("[6/6] Generating publication-grade figures in reports/figures/...")

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # Figure 1: Training Loss & Accuracy Convergence Curves
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {
        "Spectral 1D-CNN": "#2196f3",
        "Spatial 2D-CNN": "#4caf50",
        "Spatial-Spectral 3D-CNN": "#ff9800",
        "HybridSN (3D-2D CNN)": "#9c27b0"
    }

    # (a) Random Split Loss
    ax = axes[0, 0]
    for name, hist in histories["random_split"].items():
        ax.plot(hist["epoch"], hist["train_loss"], label=f"{name} (Train)", color=colors[name], linewidth=2.0)
        ax.plot(hist["epoch"], hist["test_loss"], label=f"{name} (Val)", color=colors[name], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title("(a) Regime A: Naive Random Split — Cross-Entropy Loss Convergence", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    # (b) Random Split Test Accuracy
    ax = axes[0, 1]
    for name, hist in histories["random_split"].items():
        acc_pct = [a * 100.0 for a in hist["test_acc"]]
        ax.plot(hist["epoch"], acc_pct, label=f"{name} (Peak: {max(acc_pct):.1f}%)", color=colors[name], linewidth=2.0, marker="o", markersize=4)
    ax.set_title("(b) Regime A: Naive Random Split — Test Overall Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(50, 101)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")

    # (c) Disjoint Parcel Split Loss
    ax = axes[1, 0]
    for name, hist in histories["parcel_split"].items():
        ax.plot(hist["epoch"], hist["train_loss"], label=f"{name} (Train)", color=colors[name], linewidth=2.0)
        ax.plot(hist["epoch"], hist["test_loss"], label=f"{name} (Val)", color=colors[name], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title("(c) Regime B: Disjoint Parcel Split — Cross-Entropy Loss Dynamics", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    # (d) Disjoint Parcel Split Test Accuracy
    ax = axes[1, 1]
    for name, hist in histories["parcel_split"].items():
        acc_pct = [a * 100.0 for a in hist["test_acc"]]
        ax.plot(hist["epoch"], acc_pct, label=f"{name} (Final: {acc_pct[-1]:.1f}%)", color=colors[name], linewidth=2.0, marker="s", markersize=4)
    ax.set_title("(d) Regime B: Disjoint Parcel Split — Unseen Field Parcel Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(15, 60)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")

    plt.suptitle("TerraSpectra: Deep Learning Training Dynamics Across Spatial Regimes (Indian Pines)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "day04_dl_training_curves.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: The Deep Learning Generalization Cliff Benchmark
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    models = ["Spectral 1D-CNN", "Spatial 2D-CNN", "Spatial-Spectral 3D-CNN", "HybridSN (3D-2D CNN)"]
    x = np.arange(len(models))
    width = 0.35

    oa_rnd = [benchmark_results["random_split"][m]["overall_accuracy_oa"] * 100.0 for m in models]
    oa_prc = [benchmark_results["parcel_split"][m]["overall_accuracy_oa"] * 100.0 for m in models]
    kappa_rnd = [benchmark_results["random_split"][m]["cohens_kappa"] for m in models]
    kappa_prc = [benchmark_results["parcel_split"][m]["cohens_kappa"] for m in models]

    # Bar plot 1: Overall Accuracy & Generalization Cliff
    rects1 = ax1.bar(x - width/2, oa_rnd, width, label="Regime A: Naive Random Split (Spatial Overlap)", color="#42a5f5", edgecolor="#1565c0", alpha=0.9)
    rects2 = ax1.bar(x + width/2, oa_prc, width, label="Regime B: Disjoint Parcel Split (Zero Overlap)", color="#ef5350", edgecolor="#c62828", alpha=0.9)

    ax1.set_ylabel("Overall Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Overall Accuracy (OA) — The Deep Learning Generalization Cliff", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.7)

    # Annotate drop percentages
    for i in range(len(models)):
        drop_val = oa_rnd[i] - oa_prc[i]
        rel_drop = (drop_val / oa_rnd[i]) * 100.0
        ax1.annotate(f"Cliff: -{drop_val:.1f}%\n(rel -{rel_drop:.0f}%)",
                     xy=(x[i] + width/2, oa_prc[i] + 3),
                     xytext=(x[i] + width/2, oa_prc[i] + 16),
                     ha="center", fontsize=8, fontweight="bold", color="#b71c1c",
                     arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2))

    # Bar plot 2: Cohen's Kappa Comparison
    ax2.bar(x - width/2, kappa_rnd, width, label="Regime A: Naive Random Split", color="#66bb6a", edgecolor="#2e7d32", alpha=0.9)
    ax2.bar(x + width/2, kappa_prc, width, label="Regime B: Disjoint Parcel Split", color="#ffa726", edgecolor="#ef6c00", alpha=0.9)

    ax2.set_ylabel("Cohen's Kappa Coefficient (κ)", fontsize=11, fontweight="bold")
    ax2.set_title("Inter-Rater Reliability (Cohen's Kappa κ)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", linestyle=":", alpha=0.7)

    for i in range(len(models)):
        ax2.text(x[i] - width/2, kappa_rnd[i] + 0.02, f"{kappa_rnd[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax2.text(x[i] + width/2, kappa_prc[i] + 0.02, f"{kappa_prc[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("TerraSpectra: Empirical Quantification of the Generalization Cliff across Deep Neural Architectures", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "day04_deep_learning_benchmark_cliff.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Full Scene Thematic Spatial Prediction Maps
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15)

    # 1. False Color CIR (NIR: 800nm, Red: 670nm, Green: 550nm)
    cube_norm = data_dict["cube_normalized"]
    r_band = np.clip((cube_norm[:, :, 40] - np.percentile(cube_norm[:, :, 40], 2)) / (np.percentile(cube_norm[:, :, 40], 98) - np.percentile(cube_norm[:, :, 40], 2) + 1e-6), 0, 1)
    g_band = np.clip((cube_norm[:, :, 27] - np.percentile(cube_norm[:, :, 27], 2)) / (np.percentile(cube_norm[:, :, 27], 98) - np.percentile(cube_norm[:, :, 27], 2) + 1e-6), 0, 1)
    b_band = np.clip((cube_norm[:, :, 15] - np.percentile(cube_norm[:, :, 15], 2)) / (np.percentile(cube_norm[:, :, 15], 98) - np.percentile(cube_norm[:, :, 15], 2) + 1e-6), 0, 1)
    cir_rgb = np.dstack([r_band, g_band, b_band])

    display_panels = [
        (gs[0, 0], "AVIRIS False-Color CIR (800 / 670 / 550 nm)", cir_rgb, None),
        (gs[0, 1], "Ground Truth (16 Agricultural Classes)", pred_maps["Ground Truth"], IP_CMAP),
        (gs[0, 2], f"Spectral 1D-CNN (OA: {benchmark_results['parcel_split']['Spectral 1D-CNN']['overall_accuracy_oa']*100:.1f}%)", pred_maps["Spectral 1D-CNN"], IP_CMAP),
        (gs[1, 0], f"Spatial 2D-CNN (OA: {benchmark_results['parcel_split']['Spatial 2D-CNN']['overall_accuracy_oa']*100:.1f}%)", pred_maps["Spatial 2D-CNN"], IP_CMAP),
        (gs[1, 1], f"Spatial-Spectral 3D-CNN (OA: {benchmark_results['parcel_split']['Spatial-Spectral 3D-CNN']['overall_accuracy_oa']*100:.1f}%)", pred_maps["Spatial-Spectral 3D-CNN"], IP_CMAP),
        (gs[1, 2], f"HybridSN (3D-2D CNN) (OA: {benchmark_results['parcel_split']['HybridSN (3D-2D CNN)']['overall_accuracy_oa']*100:.1f}%)", pred_maps["HybridSN (3D-2D CNN)"], IP_CMAP)
    ]

    for grid_pos, title, img_data, cmap in display_panels:
        ax = fig.add_subplot(grid_pos)
        if cmap is None:
            ax.imshow(img_data)
        else:
            ax.imshow(img_data, cmap=cmap, vmin=0, vmax=16)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # Add shared categorical legend for classes 1..16 at bottom
    patches = [mpatches.Patch(color=IP_HEX_COLORS[c], label=f"{c}: {INDIAN_PINES_CLASSES[c]}") for c in range(1, 17)]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("TerraSpectra: Full-Scene Thematic Classification Maps Under Disjoint Parcel Evaluation", fontsize=14, fontweight="bold", y=0.98)
    fig3_path = os.path.join(FIG_DIR, "day04_full_scene_thematic_maps.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig3_path}")

    # -------------------------------------------------------------------------
    # Figure 4: Receptive Field Ablation & Confusion Matrix
    # -------------------------------------------------------------------------
    fig, (ax_abl, ax_eff) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Patch Window Ablation
    k_vals = ablation_df["spatial_k"].values
    oa_vals = ablation_df["overall_accuracy_oa"].values * 100.0
    aa_vals = ablation_df["average_accuracy_aa"].values * 100.0

    ax_abl.plot(k_vals, oa_vals, marker="o", color="#1976d2", linewidth=2.5, markersize=8, label="Overall Accuracy (OA %)")
    ax_abl.plot(k_vals, aa_vals, marker="s", color="#388e3c", linewidth=2.5, markersize=8, label="Average Accuracy (AA %)")
    ax_abl.set_xticks(k_vals)
    ax_abl.set_xticklabels([f"{k}x{k}\n({k*k} px)" for k in k_vals], fontsize=10, fontweight="bold")
    ax_abl.set_xlabel("Spatial Patch Context Window (k x k)", fontsize=11, fontweight="bold")
    ax_abl.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax_abl.set_title("(a) Spatial Receptive Field Sensitivity Analysis on Disjoint Parcels", fontsize=12, fontweight="bold", pad=10)
    ax_abl.grid(True, linestyle=":", alpha=0.7)
    ax_abl.legend(fontsize=10, loc="lower right")

    for k, oa in zip(k_vals, oa_vals):
        ax_abl.annotate(f"{oa:.1f}%", xy=(k, oa), xytext=(k, oa + 1.2), ha="center", fontsize=9, fontweight="bold", color="#1976d2")

    # Right: Parameter Efficiency vs Disjoint Accuracy Scatter
    arch_names = list(benchmark_results["parcel_split"].keys())
    params = [benchmark_results["parcel_split"][m]["parameter_count"] for m in arch_names]
    accuracies = [benchmark_results["parcel_split"][m]["overall_accuracy_oa"] * 100.0 for m in arch_names]
    latencies = [benchmark_results["parcel_split"][m]["inference_ms_per_sample"] for m in arch_names]

    scatter = ax_eff.scatter(params, accuracies, s=[l * 400 + 100 for l in latencies],
                             c=["#2196f3", "#4caf50", "#ff9800", "#9c27b0"], alpha=0.85, edgecolors="#222222", linewidth=1.5)

    ax_eff.set_xscale("log")
    ax_eff.set_xlabel("Model Parameter Count (Log Scale)", fontsize=11, fontweight="bold")
    ax_eff.set_ylabel("Disjoint Parcel Split OA (%)", fontsize=11, fontweight="bold")
    ax_eff.set_title("(b) Model Parameter Efficiency vs Field Generalization Accuracy\n(Bubble size proportional to inference latency)", fontsize=12, fontweight="bold", pad=10)
    ax_eff.grid(True, linestyle=":", alpha=0.7)

    for i, name in enumerate(arch_names):
        ax_eff.annotate(f"{name}\n({accuracies[i]:.1f}%, {latencies[i]:.2f}ms)",
                        xy=(params[i], accuracies[i]),
                        xytext=(params[i] * 1.15, accuracies[i] - 0.5),
                        fontsize=8, fontweight="bold")

    plt.suptitle("TerraSpectra: Architectural Trade-Offs & Spatial Window Optimization", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, "day04_patch_size_ablation_and_confusion.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  -> Saved {fig4_path}")


# -----------------------------------------------------------------------------
# Data & Metadata Export Engine
# -----------------------------------------------------------------------------
def export_processed_metrics(benchmark_results, training_histories):
    """
    Exports clean JSON and CSV summaries to data/processed/.
    """
    # 1. Clean JSON benchmark
    clean_json = {
        "random_split": {},
        "parcel_split": {},
        "generalization_cliff": benchmark_results["generalization_cliff"]
    }
    for m in benchmark_results["random_split"]:
        clean_json["random_split"][m] = {k: v for k, v in benchmark_results["random_split"][m].items() if k not in ["history", "trained_model"]}
    for m in benchmark_results["parcel_split"]:
        clean_json["parcel_split"][m] = {k: v for k, v in benchmark_results["parcel_split"][m].items() if k not in ["history", "trained_model"]}

    json_path = os.path.join(PROC_DATA_DIR, "day04_dl_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(clean_json, f, indent=4)
    print(f"  -> Saved benchmark results to {json_path}")

    # 2. Training Histories JSON
    hist_path = os.path.join(PROC_DATA_DIR, "day04_dl_training_histories.json")
    with open(hist_path, "w") as f:
        json.dump(training_histories, f, indent=4)
    print(f"  -> Saved training histories to {hist_path}")

    # 3. Class-by-Class Comparison CSV
    class_rows = []
    for c_idx in range(1, 17):
        c_name = INDIAN_PINES_CLASSES[c_idx]
        row = {"class_id": c_idx, "class_name": c_name}
        for m in benchmark_results["parcel_split"]:
            metrics = benchmark_results["parcel_split"][m]["per_class_metrics"].get(c_name, {})
            row[f"{m}_recall"] = metrics.get("recall", 0.0)
            row[f"{m}_f1"] = metrics.get("f1_score", 0.0)
        class_rows.append(row)

    class_df = pd.DataFrame(class_rows)
    class_csv_path = os.path.join(PROC_DATA_DIR, "day04_dl_per_class_performance.csv")
    class_df.to_csv(class_csv_path, index=False)
    print(f"  -> Saved per-class performance table to {class_csv_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()
    print("=" * 80)
    print("TerraSpectra: Day 4 - Spatial-Spectral Deep Learning Architectures Benchmark")
    print("1D-CNN vs 2D-CNN vs 3D-CNN & HybridSN | Generalization Cliff & Receptive Fields")
    print("=" * 80)

    # 1. Load data & PCA
    data_dict = load_and_preprocess_hyperspectral_data()

    # 2. Evaluation splits
    splits = construct_evaluation_splits(data_dict["gt"], data_dict["coords"], data_dict["labels"])

    # 3. Deep learning benchmark
    benchmark_results, training_histories, trained_models = execute_deep_learning_benchmark_suite(data_dict, splits)

    # 4. Patch size ablation
    ablation_df = run_patch_size_ablation(data_dict, splits)

    # 5. Full scene spatial inference
    pred_maps = generate_full_scene_prediction_maps(data_dict, trained_models)

    # 6. Figures & exports
    generate_publication_figures(data_dict, training_histories, benchmark_results, pred_maps, ablation_df)
    export_processed_metrics(benchmark_results, training_histories)

    total_duration = time.time() - start_time
    print("=" * 80)
    print(f"Day 4 Deep Learning Pipeline Completed Successfully in {total_duration / 60.0:.2f} minutes.")
    print("=" * 80)


if __name__ == "__main__":
    main()
