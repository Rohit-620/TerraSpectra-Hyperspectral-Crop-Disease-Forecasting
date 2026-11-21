"""
TerraSpectra: Day 5 - Hyperspectral Vision Transformers, Self-Supervised Masked Autoencoders (HS-MAE),
and Cross-Scene Domain Adaptation (Indian Pines <-> Salinas Valley)
Author: Rohit (TerraSpectra ML Team)
Working Branch: Rohit-EDA-Analysis
Milestone: Day 5 - Foundation Models, Attention Rollout, and Domain Transfer for Hyperspectral Remote Sensing
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
# Deep Learning Architecture Suite: Transformers & Foundation Models
# -----------------------------------------------------------------------------

class HyperspectralMAE(nn.Module):
    """
    Hyperspectral Masked Autoencoder (HS-MAE) for Self-Supervised Pretraining.
    Ingests spatial-spectral patches (batch_size, num_tokens=49, channels=30).
    Randomly masks 70% of tokens and trains a Transformer Encoder-Decoder to reconstruct
    the masked spectral-spatial volume via Mean Squared Error (MSE).
    """
    def __init__(self, in_channels=30, num_tokens=49, embed_dim=64, decoder_embed_dim=64,
                 mask_ratio=0.70, enc_layers=3, dec_layers=1, nhead=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # Encoder: Linear patch projection + positional embeddings
        self.proj = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.enc_norm = nn.LayerNorm(embed_dim)

        # Decoder: Shared mask token + decoder positional embeddings
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.dec_pos_embed = nn.Parameter(torch.randn(1, num_tokens, decoder_embed_dim) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        self.dec_norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, in_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.pred.weight)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, x):
        B = x.shape[0]
        tokens = self.proj(x) + self.pos_embed
        x_vis, mask, ids_restore = self.random_masking(tokens, self.mask_ratio)

        # Encoder pass on visible tokens only
        latent = self.encoder(x_vis)
        latent = self.enc_norm(latent)

        # Decoder pass: reconstitute full sequence with mask tokens
        mask_tokens = self.mask_token.repeat(B, self.num_tokens - x_vis.shape[1], 1)
        x_full = torch.cat([latent, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_embed_dim))
        x_full = x_full + self.dec_pos_embed

        dec_out = self.decoder(x_full)
        dec_out = self.dec_norm(dec_out)
        pred = self.pred(dec_out)

        # Calculate MSE loss strictly over masked positions
        loss = ((pred - x) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss, pred, mask


class MAEClassifier(nn.Module):
    """
    Downstream Crop Classifier initializing from a pretrained HS-MAE Encoder.
    Evaluates representation transfer from unlabeled agricultural cubes.
    """
    def __init__(self, pretrained_mae, num_classes=16, freeze_encoder=False):
        super().__init__()
        self.proj = pretrained_mae.proj
        self.pos_embed = pretrained_mae.pos_embed
        self.encoder = pretrained_mae.encoder
        self.enc_norm = pretrained_mae.enc_norm

        if freeze_encoder:
            for p in self.proj.parameters():
                p.requires_grad = False
            self.pos_embed.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.enc_norm.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(pretrained_mae.embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        tokens = self.proj(x) + self.pos_embed
        latent = self.encoder(tokens)
        latent = self.enc_norm(latent)
        # Global Average Pooling across spatial-spectral tokens
        pooled = latent.mean(dim=1)
        return self.classifier(pooled)


class SpectralFormer(nn.Module):
    """
    Spectral-Spatial Vision Transformer (Hong et al., IEEE TGRS 2021).
    Ingests spatial-spectral patches, prepends a learnable [CLS] token,
    and models long-range multi-head self-attention across tokens.
    Provides explicit attention rollout extraction for biophysical band attribution.
    """
    def __init__(self, in_channels=30, num_tokens=49, embed_dim=64, num_layers=3, num_heads=4, num_classes=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.proj = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, embed_dim) * 0.02)

        # Transformer blocks with accessible attention matrices
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha": nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True),
                "norm1": nn.LayerNorm(embed_dim),
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, embed_dim),
                    nn.Dropout(0.1)
                ),
                "norm2": nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer["mlp"][0].weight)
            nn.init.xavier_uniform_(layer["mlp"][3].weight)

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        tokens = self.proj(x)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x_seq = torch.cat([cls_tokens, tokens], dim=1) + self.pos_embed

        attn_weights = []
        for layer in self.layers:
            # Pre-LN MHA with residual connection
            norm_x = layer["norm1"](x_seq)
            if return_attn:
                attn_out, attn_map = layer["mha"](norm_x, norm_x, norm_x, need_weights=True, average_attn_weights=False)
                attn_weights.append(attn_map)
            else:
                attn_out, _ = layer["mha"](norm_x, norm_x, norm_x, need_weights=False)
            x_seq = x_seq + attn_out
            # Pre-LN MLP with residual connection
            norm_x2 = layer["norm2"](x_seq)
            mlp_out = layer["mlp"](norm_x2)
            x_seq = x_seq + mlp_out

        x_norm = self.norm(x_seq)
        cls_out = x_norm[:, 0]
        logits = self.head(cls_out)

        if return_attn:
            return logits, attn_weights
        return logits


class HybridSN(nn.Module):
    """
    Day 4 SOTA Convolutional Reference: Hybrid 3D-2D Spatial-Spectral Network.
    Enables direct architectural comparison between Convolutional and Attention paradigms.
    """
    def __init__(self, in_channels=1, num_classes=16):
        super().__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        # Reshape to 2D
        # For input (1, 30, 7, 7):
        # Conv3D (5,3,3) -> (8, 26, 5, 5)
        # Conv3D (5,3,3) -> (16, 22, 3, 3)
        # Reshaped to 2D: (16 * 22 = 352, 3, 3)
        self.conv2d = nn.Sequential(
            nn.Conv2d(16 * 22, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x is (B, 49, 30) -> reshape to (B, 1, 30, 7, 7)
        B = x.shape[0]
        x_cube = x.view(B, 7, 7, 30).permute(0, 3, 1, 2).unsqueeze(1)
        f1 = self.conv3d_1(x_cube)
        f2 = self.conv3d_2(f1)
        b, c, d, h, w = f2.shape
        f2_2d = f2.view(b, c * d, h, w)
        f3 = self.conv2d(f2_2d)
        flat = f3.view(b, -1)
        return self.classifier(flat)


# -----------------------------------------------------------------------------
# Correlation Alignment (CORAL) Domain Adaptation Loss
# -----------------------------------------------------------------------------
def compute_coral_loss(source, target):
    """
    Computes Correlation Alignment (CORAL) loss between source and target domain features.
    Aligns second-order statistics (covariance matrices) across agricultural domains.
    L_CORAL = (1 / 4d^2) * || C_S - C_T ||_F^2
    """
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # Source covariance
    ones_s = torch.ones(1, ns, device=source.device)
    tmp_s = ones_s @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1 + 1e-8)

    # Target covariance
    ones_t = torch.ones(1, nt, device=target.device)
    tmp_t = ones_t @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1 + 1e-8)

    loss = (cs - ct).pow(2).sum() / (4 * d * d)
    return loss


class DomainAdaptationTransformer(nn.Module):
    """
    Transformer Architecture for Cross-Scene Domain Adaptation.
    Maps Source (Salinas) and Target (Indian Pines) to a shared embedding space,
    optimizing joint cross-entropy and CORAL covariance alignment.
    """
    def __init__(self, in_channels=30, num_tokens=49, embed_dim=64, num_classes=16):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def extract_features(self, x):
        B = x.shape[0]
        tokens = self.proj(x)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x_seq = torch.cat([cls_tokens, tokens], dim=1) + self.pos_embed
        out = self.transformer(x_seq)
        out = self.norm(out)
        return out[:, 0]  # CLS feature representation

    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits, feat


# -----------------------------------------------------------------------------
# Data Loading, PCA Reduction & Patch Extraction Engine
# -----------------------------------------------------------------------------
def load_and_prepare_hyperspectral_scenes():
    """
    Ingests Indian Pines and Salinas Valley datasets.
    Performs PCA-30 reduction, extracts spatial-spectral patches (7x7),
    and constructs unlabeled pretraining sets alongside labeled benchmark splits.
    """
    print("[1/8] Ingesting hyperspectral scenes (Indian Pines & Salinas Valley)...")
    
    # 1. Load Indian Pines
    ip_cube_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_corrected.mat"))
    ip_gt_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Indian_pines_gt.mat"))
    ip_cube = [ip_cube_mat[k] for k in ip_cube_mat if not k.startswith("__")][0]
    ip_gt = [ip_gt_mat[k] for k in ip_gt_mat if not k.startswith("__")][0]

    # 2. Load Salinas Valley
    sal_cube_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Salinas_corrected.mat"))
    sal_gt_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "Salinas_gt.mat"))
    sal_cube = [sal_cube_mat[k] for k in sal_cube_mat if not k.startswith("__")][0]
    sal_gt = [sal_gt_mat[k] for k in sal_gt_mat if not k.startswith("__")][0]

    print(f"  -> Indian Pines scene: {ip_cube.shape} | Labeled pixels: {np.sum(ip_gt > 0)}")
    print(f"  -> Salinas Valley scene: {sal_cube.shape} | Labeled pixels: {np.sum(sal_gt > 0)}")

    # Normalize to [0, 1] reflectance
    ip_norm = ip_cube.astype(np.float32) / 10000.0
    sal_norm = sal_cube.astype(np.float32) / 10000.0

    # Fit PCA-30 on Indian Pines
    H_ip, W_ip, B_ip = ip_cube.shape
    pca_ip = PCA(n_components=30, random_state=42)
    ip_pca = pca_ip.fit_transform(ip_norm.reshape(-1, B_ip)).reshape(H_ip, W_ip, 30)
    explained_ip = float(np.sum(pca_ip.explained_variance_ratio_))

    # Fit PCA-30 on Salinas Valley
    H_sal, W_sal, B_sal = sal_cube.shape
    pca_sal = PCA(n_components=30, random_state=42)
    sal_pca = pca_sal.fit_transform(sal_norm.reshape(-1, B_sal)).reshape(H_sal, W_sal, 30)

    print(f"  -> Indian Pines PCA-30 explained variance: {explained_ip * 100.0:.2f}%")

    # -------------------------------------------------------------------------
    # Spatial Patch Extraction (Window Size k=7)
    # -------------------------------------------------------------------------
    k = 7
    m = k // 2
    padded_ip = np.pad(ip_pca, ((m, m), (m, m), (0, 0)), mode="reflect")

    # Labeled samples extraction
    labeled_mask = (ip_gt > 0)
    coords = np.argwhere(labeled_mask)
    labels = ip_gt[labeled_mask] - 1  # 0-indexed [0..15]
    n_samples = len(labels)

    X_ip_patches = np.zeros((n_samples, k * k, 30), dtype=np.float32)
    for i, (r, c) in enumerate(coords):
        patch = padded_ip[r:r + k, c:c + k, :]
        X_ip_patches[i] = patch.reshape(-1, 30)

    print(f"  -> Extracted {n_samples} labeled spatial-spectral patches of size {k}x{k}x30.")

    # -------------------------------------------------------------------------
    # Unlabeled Background Extraction for HS-MAE Pre-training
    # -------------------------------------------------------------------------
    unlabeled_mask = (ip_gt == 0)
    unlabeled_coords = np.argwhere(unlabeled_mask)
    np.random.seed(42)
    sample_unlabeled_idx = np.random.choice(len(unlabeled_coords), size=min(4000, len(unlabeled_coords)), replace=False)
    selected_unlabeled_coords = unlabeled_coords[sample_unlabeled_idx]

    X_unlabeled_patches = np.zeros((len(selected_unlabeled_coords), k * k, 30), dtype=np.float32)
    for i, (r, c) in enumerate(selected_unlabeled_coords):
        patch = padded_ip[r:r + k, c:c + k, :]
        X_unlabeled_patches[i] = patch.reshape(-1, 30)

    # Combine labeled and unlabeled patches for Self-Supervised pretraining
    X_mae_pretrain = np.vstack([X_ip_patches, X_unlabeled_patches])
    np.random.seed(42)
    np.random.shuffle(X_mae_pretrain)
    print(f"  -> Constructed HS-MAE Self-Supervised pretraining pool: {len(X_mae_pretrain)} total patches ({len(X_unlabeled_patches)} unlabeled background).")

    # -------------------------------------------------------------------------
    # Salinas Patch Extraction for Domain Adaptation
    # -------------------------------------------------------------------------
    padded_sal = np.pad(sal_pca, ((m, m), (m, m), (0, 0)), mode="reflect")
    sal_labeled_mask = (sal_gt > 0)
    sal_coords = np.argwhere(sal_labeled_mask)
    sal_labels = sal_gt[sal_labeled_mask] - 1

    # Sample balanced subset from Salinas
    np.random.seed(42)
    sal_sample_idx = np.random.choice(len(sal_labels), size=min(6000, len(sal_labels)), replace=False)
    X_sal_patches = np.zeros((len(sal_sample_idx), k * k, 30), dtype=np.float32)
    y_sal_samples = sal_labels[sal_sample_idx]

    for i, idx in enumerate(sal_sample_idx):
        r, c = sal_coords[idx]
        patch = padded_sal[r:r + k, c:c + k, :]
        X_sal_patches[i] = patch.reshape(-1, 30)

    print(f"  -> Extracted {len(X_sal_patches)} Salinas domain adaptation patches.")

    return {
        "ip_raw": ip_cube,
        "ip_norm": ip_norm,
        "ip_pca": ip_pca,
        "ip_gt": ip_gt,
        "coords": coords,
        "labels": labels,
        "X_ip_patches": X_ip_patches,
        "X_mae_pretrain": X_mae_pretrain,
        "X_sal_patches": X_sal_patches,
        "y_sal": y_sal_samples,
        "explained_ip": explained_ip
    }


def construct_disjoint_parcel_splits(gt, coords, labels, random_seed=42):
    """
    Constructs both:
    1. Regime A: Random Split (80% train / 20% test stratified)
    2. Regime B: Disjoint Parcel Split (intact connected-component parcel isolation)
    """
    print("[2/8] Generating evaluation splits (Random vs Spatial Disjoint Parcel)...")
    H, W = gt.shape
    n_samples = len(labels)

    # 1. Random Split
    idx_rnd_train, idx_rnd_test = train_test_split(
        np.arange(n_samples), test_size=0.20, random_state=random_seed, stratify=labels
    )

    # 2. Disjoint Parcel Split
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
# Stage 1 & 2: HS-MAE Pre-training & Reconstruction Inspection
# -----------------------------------------------------------------------------
def pretrain_hyperspectral_mae(X_pretrain, epochs=12, batch_size=128, lr=1e-3, seed=42):
    """
    Self-Supervised Pre-training of Hyperspectral Masked Autoencoder (HS-MAE).
    Optimizes masked MSE loss across visible/masked tokens without human labels.
    """
    print("[3/8] Executing Self-Supervised HS-MAE Pre-training (70% masking ratio)...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(torch.from_numpy(X_pretrain))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mae = HyperspectralMAE(in_channels=30, num_tokens=49, embed_dim=64, mask_ratio=0.70)
    optimizer = optim.AdamW(mae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"epoch": [], "loss": [], "lr": []}
    start_time = time.time()

    for ep in range(1, epochs + 1):
        mae.train()
        running_loss = 0.0
        total_samples = 0

        for (bx,) in loader:
            optimizer.zero_grad()
            loss, pred, mask = mae(bx)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(bx)
            total_samples += len(bx)

        scheduler.step()
        ep_loss = running_loss / total_samples
        history["epoch"].append(ep)
        history["loss"].append(round(ep_loss, 6))
        history["lr"].append(round(float(scheduler.get_last_lr()[0]), 6))

        if ep % 2 == 0 or ep == epochs:
            print(f"  -> Epoch {ep:02d}/{epochs:02d} | MAE Masked Reconstruction MSE: {ep_loss:.6f}")

    total_pretrain_time = time.time() - start_time
    print(f"  -> HS-MAE Pre-training finalized in {total_pretrain_time:.2f} seconds.")
    return mae, history, total_pretrain_time


# -----------------------------------------------------------------------------
# Stage 3, 4, 5: Supervised Benchmark Engine (ViT vs MAE-FT vs HybridSN)
# -----------------------------------------------------------------------------
def train_and_evaluate_classifier(model, model_name, X_data, y_data, splits, regime_name,
                                  epochs=10, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=42):
    """
    Trains and benchmarks a classifier under specified split regime (Random vs Parcel).
    Calculates OA, AA, Kappa, Macro-F1, per-class metrics, and inference latency.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx = splits[regime_name]["train"]
    test_idx = splits[regime_name]["test"]

    X_tr, y_tr = torch.from_numpy(X_data[train_idx]), torch.from_numpy(y_data[train_idx])
    X_te, y_te = torch.from_numpy(X_data[test_idx]), torch.from_numpy(y_data[test_idx])

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start_fit = time.time()

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
        tr_loss = running_loss / total_tr
        tr_acc = correct_tr / total_tr

        # Evaluation
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

        te_loss = running_te_loss / total_te
        te_acc = correct_te / total_te

        history["epoch"].append(ep)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["test_loss"].append(round(te_loss, 4))
        history["test_acc"].append(round(te_acc, 4))

    fit_time = time.time() - start_fit

    # Inference profiling
    model.eval()
    all_preds = []
    all_targets = []
    start_infer = time.time()
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_targets.extend(by.cpu().numpy())
    infer_time = time.time() - start_infer
    ms_per_sample = (infer_time / len(test_idx)) * 1000.0

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    oa = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Average Accuracy (Mean Recall)
    unique_classes = np.unique(y_data)
    recalls = recall_score(all_targets, all_preds, labels=unique_classes, average=None, zero_division=0)
    aa = np.mean(recalls)

    # Per-class metrics
    precisions = precision_score(all_targets, all_preds, labels=unique_classes, average=None, zero_division=0)
    f1s = f1_score(all_targets, all_preds, labels=unique_classes, average=None, zero_division=0)

    per_class_dict = {}
    for i, cls_id in enumerate(unique_classes):
        cls_name = INDIAN_PINES_CLASSES.get(cls_id + 1, f"Class_{cls_id+1}")
        per_class_dict[cls_name] = {
            "precision": round(float(precisions[i]), 4),
            "recall": round(float(recalls[i]), 4),
            "f1_score": round(float(f1s[i]), 4)
        }

    cm = confusion_matrix(all_targets, all_preds, labels=unique_classes)

    res = {
        "model_name": model_name,
        "regime": regime_name,
        "overall_accuracy_oa": round(float(oa), 4),
        "average_accuracy_aa": round(float(aa), 4),
        "cohens_kappa": round(float(kappa), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "fit_time_seconds": round(float(fit_time), 2),
        "inference_ms_per_sample": round(float(ms_per_sample), 4),
        "parameter_count": param_count,
        "per_class_metrics": per_class_dict,
        "confusion_matrix": cm.tolist()
    }

    print(f"    [{model_name}] {regime_name.upper()} -> OA: {oa * 100:.2f}% | AA: {aa * 100:.2f}% | Kappa: {kappa:.4f} | Macro-F1: {macro_f1:.4f} ({fit_time:.1f}s)")
    return res, history, model


# -----------------------------------------------------------------------------
# Stage 6: Cross-Scene Domain Adaptation (Salinas -> Indian Pines)
# -----------------------------------------------------------------------------
def run_cross_scene_domain_adaptation(X_sal, y_sal, X_ip, y_ip, epochs=10, batch_size=128, lr=1e-3, seed=42):
    """
    Evaluates cross-scene generalization from Salinas Valley (Source) to Indian Pines (Target).
    Compares:
    1. Zero-shot transfer (Source Only)
    2. Domain Adaptation via CORAL Covariance Alignment
    3. Target Supervised Oracle
    """
    print("[6/8] Executing Cross-Scene Domain Adaptation (Salinas -> Indian Pines)...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Source Only Model
    source_model = DomainAdaptationTransformer(in_channels=30, num_tokens=49, embed_dim=64, num_classes=16)
    optimizer_s = optim.Adam(source_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    sal_loader = DataLoader(TensorDataset(torch.from_numpy(X_sal), torch.from_numpy(y_sal)), batch_size=batch_size, shuffle=True)
    ip_test_loader = DataLoader(TensorDataset(torch.from_numpy(X_ip), torch.from_numpy(y_ip)), batch_size=256, shuffle=False)

    print("  -> Training Source-Only model on Salinas Valley...")
    for ep in range(1, epochs + 1):
        source_model.train()
        for bx, by in sal_loader:
            optimizer_s.zero_grad()
            logits, _ = source_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer_s.step()

    # Zero-shot evaluation on Indian Pines
    source_model.eval()
    preds_zero_shot = []
    with torch.no_grad():
        for bx, by in ip_test_loader:
            logits, _ = source_model(bx)
            preds_zero_shot.extend(logits.argmax(dim=1).cpu().numpy())
    oa_zero_shot = accuracy_score(y_ip, preds_zero_shot)

    # 2. Domain Adaptation with CORAL Alignment
    print("  -> Training CORAL Domain-Adapted Transformer...")
    coral_model = DomainAdaptationTransformer(in_channels=30, num_tokens=49, embed_dim=64, num_classes=16)
    optimizer_coral = optim.Adam(coral_model.parameters(), lr=lr)

    # Target unlabeled loader
    ip_unlabeled_loader = DataLoader(TensorDataset(torch.from_numpy(X_ip)), batch_size=batch_size, shuffle=True)
    lambda_coral = 0.5

    coral_history = []
    for ep in range(1, epochs + 1):
        coral_model.train()
        iter_target = iter(ip_unlabeled_loader)
        running_cls_loss = 0.0
        running_coral_loss = 0.0

        for bx_s, by_s in sal_loader:
            try:
                (bx_t,) = next(iter_target)
            except StopIteration:
                iter_target = iter(ip_unlabeled_loader)
                (bx_t,) = next(iter_target)

            optimizer_coral.zero_grad()
            logits_s, feat_s = coral_model(bx_s)
            _, feat_t = coral_model(bx_t)

            loss_cls = criterion(logits_s, by_s)
            loss_c = compute_coral_loss(feat_s, feat_t)
            total_loss = loss_cls + lambda_coral * loss_c

            total_loss.backward()
            optimizer_coral.step()

            running_cls_loss += loss_cls.item()
            running_coral_loss += loss_c.item()

        coral_history.append((running_cls_loss, running_coral_loss))

    # Evaluate CORAL model on Indian Pines
    coral_model.eval()
    preds_coral = []
    feats_source = []
    feats_target = []
    with torch.no_grad():
        for bx, by in ip_test_loader:
            logits, feat = coral_model(bx)
            preds_coral.extend(logits.argmax(dim=1).cpu().numpy())
            feats_target.append(feat.cpu().numpy())
        for bx, _ in sal_loader:
            _, feat = coral_model(bx)
            feats_source.append(feat.cpu().numpy())
            if len(feats_source) >= 10:
                break

    oa_coral = accuracy_score(y_ip, preds_coral)
    feats_source = np.vstack(feats_source)
    feats_target = np.vstack(feats_target)

    print(f"  -> Cross-Scene Zero-Shot OA: {oa_zero_shot * 100.0:.2f}%")
    print(f"  -> Cross-Scene CORAL-Adapted OA: {oa_coral * 100.0:.2f}% (Gain: +{(oa_coral - oa_zero_shot) * 100.0:.2f}%)")

    return {
        "zero_shot_oa": round(float(oa_zero_shot), 4),
        "coral_oa": round(float(oa_coral), 4),
        "gain_percent": round(float((oa_coral - oa_zero_shot) * 100.0), 2),
        "feats_source": feats_source[:500],
        "feats_target": feats_target[:500]
    }


# -----------------------------------------------------------------------------
# Stage 7: Biophysical Spectral Attention Rollout & Wavelength Attribution
# -----------------------------------------------------------------------------
def extract_spectral_attention_rollout(vit_model, X_sample):
    """
    Extracts multi-head self-attention weights from SpectralFormer layers,
    computes attention rollout from the [CLS] token across spatial-spectral tokens,
    and attributes spectral importance to biophysical wavelength regimes.
    """
    print("[7/8] Extracting Multi-Head Attention Rollout & Biophysical Wavelength Attribution...")
    vit_model.eval()
    with torch.no_grad():
        bx = torch.from_numpy(X_sample[:100])
        _, attn_layers = vit_model(bx, return_attn=True)

    # attn_layers is list of tensors: each (B, num_heads, 50, 50)
    # Average across batch and heads
    avg_attns = [layer.mean(dim=(0, 1)).cpu().numpy() for layer in attn_layers]

    # Attention Rollout: A_rollout = A_1 * A_2 * ... * A_L
    eye = np.eye(50)
    rollout = eye
    for A in avg_attns:
        A_norm = 0.5 * A + 0.5 * eye
        A_norm = A_norm / A_norm.sum(axis=-1, keepdims=True)
        rollout = np.matmul(A_norm, rollout)

    # Spatial token attention from [CLS] token (indices 1 to 49)
    cls_spatial_attn = rollout[0, 1:].reshape(7, 7)
    cls_spatial_attn = cls_spatial_attn / cls_spatial_attn.sum()

    # Attribute token importance to PCA components
    # Projected weight matrix (embed_dim x 30)
    W_proj = vit_model.proj.weight.detach().cpu().numpy()  # (64, 30)
    component_importance = np.abs(W_proj).mean(axis=0)
    component_importance = component_importance / component_importance.sum()

    print("  -> Extracted 3-layer Attention Rollout matrix (50x50).")
    print(f"  -> Top-3 dominant spectral components: PC{component_importance.argsort()[::-1][:3] + 1}")

    return {
        "spatial_attn_map": cls_spatial_attn,
        "component_importance": component_importance,
        "raw_rollout": rollout
    }


# -----------------------------------------------------------------------------
# Stage 8: Scientific Visualization & Reporting Pipeline
# -----------------------------------------------------------------------------
def generate_publication_figures(mae_history, mae_model, X_sample_patch, benchmark_results,
                                 attn_data, domain_results):
    """
    Generates 4 high-resolution publication-grade multi-panel figures:
    1. HS-MAE Reconstruction & Masking Topology
    2. ViT vs MAE vs CNN Generalization Cliff Benchmark
    3. Spectral Attention Rollout & Biophysical Wavelength Attribution
    4. Cross-Scene Domain Adaptation (Salinas -> Indian Pines)
    """
    print("[8/8] Rendering publication-quality multi-panel scientific figures...")

    # Set publication aesthetic
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # Figure 1: HS-MAE Masking & Reconstruction Topology
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs1 = GridSpec(2, 3, figure=fig1)

    # Prepare single patch for visual demonstration
    mae_model.eval()
    with torch.no_grad():
        test_patch = torch.from_numpy(X_sample_patch[0:1])
        loss, pred, mask = mae_model(test_patch)

    orig_patch_2d = test_patch[0].numpy().reshape(7, 7, 30)[:, :, :3]
    orig_patch_2d = (orig_patch_2d - orig_patch_2d.min()) / (orig_patch_2d.max() - orig_patch_2d.min() + 1e-8)

    mask_2d = mask[0].numpy().reshape(7, 7)
    masked_patch_2d = orig_patch_2d.copy()
    masked_patch_2d[mask_2d == 1] = 0.05  # Blackout masked tokens

    recon_patch_2d = pred[0].detach().numpy().reshape(7, 7, 30)[:, :, :3]
    recon_patch_2d = (recon_patch_2d - recon_patch_2d.min()) / (recon_patch_2d.max() - recon_patch_2d.min() + 1e-8)

    diff_map = np.abs(orig_patch_2d - recon_patch_2d).mean(axis=-1)

    # Panel A: Original Patch
    ax1_a = fig1.add_subplot(gs1[0, 0])
    ax1_a.imshow(orig_patch_2d)
    ax1_a.set_title("(A) Ground Truth Spectral Patch ($7 \\times 7$)", fontsize=11, fontweight="bold")
    ax1_a.axis("off")

    # Panel B: 70% Masked Patch
    ax1_b = fig1.add_subplot(gs1[0, 1])
    ax1_b.imshow(masked_patch_2d)
    ax1_b.set_title("(B) 70% Volumetric Token Masking", fontsize=11, fontweight="bold")
    ax1_b.axis("off")

    # Panel C: Reconstructed Patch
    ax1_c = fig1.add_subplot(gs1[0, 2])
    ax1_c.imshow(recon_patch_2d)
    ax1_c.set_title("(C) HS-MAE Reconstructed Patch", fontsize=11, fontweight="bold")
    ax1_c.axis("off")

    # Panel D: Residual Error Map
    ax1_d = fig1.add_subplot(gs1[1, 0])
    im_d = ax1_d.imshow(diff_map, cmap="inferno")
    ax1_d.set_title("(D) Reconstruction Absolute Error Map", fontsize=11, fontweight="bold")
    ax1_d.axis("off")
    fig1.colorbar(im_d, ax=ax1_d, fraction=0.046, pad=0.04)

    # Panel E: Pretraining Loss Convergence
    ax1_e = fig1.add_subplot(gs1[1, 1:3])
    ax1_e.plot(mae_history["epoch"], mae_history["loss"], "o-", color="#1f77b4", lw=2.2, label="Self-Supervised MSE Loss")
    ax1_e.set_title("(E) HS-MAE Pre-training Convergence (Unlabeled Agricultural Pool)", fontsize=11, fontweight="bold")
    ax1_e.set_xlabel("Pre-training Epoch", fontsize=10)
    ax1_e.set_ylabel("Masked Token Reconstruction MSE", fontsize=10)
    ax1_e.grid(True, linestyle="--", alpha=0.5)
    ax1_e.legend(frameon=True, facecolor="#f8f9fa")

    fig1_path = os.path.join(FIG_DIR, "day05_mae_reconstruction_and_masking.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  -> Saved Figure 1: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: The Transformer Generalization Cliff Benchmark
    # -------------------------------------------------------------------------
    fig2, (ax2_a, ax2_b) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    models = ["HybridSN", "SpectralFormer", "HS-MAE (Fine-Tuned)"]
    x = np.arange(len(models))
    width = 0.35

    oa_rnd = [
        benchmark_results["random_split"]["HybridSN"]["overall_accuracy_oa"] * 100,
        benchmark_results["random_split"]["SpectralFormer"]["overall_accuracy_oa"] * 100,
        benchmark_results["random_split"]["HS-MAE (Fine-Tuned)"]["overall_accuracy_oa"] * 100,
    ]
    oa_prc = [
        benchmark_results["parcel_split"]["HybridSN"]["overall_accuracy_oa"] * 100,
        benchmark_results["parcel_split"]["SpectralFormer"]["overall_accuracy_oa"] * 100,
        benchmark_results["parcel_split"]["HS-MAE (Fine-Tuned)"]["overall_accuracy_oa"] * 100,
    ]

    rects1 = ax2_a.bar(x - width/2, oa_rnd, width, label="Regime A: Naive Random Split (80/20)", color="#2b5c8f", edgecolor="#1a365d")
    rects2 = ax2_a.bar(x + width/2, oa_prc, width, label="Regime B: Disjoint Parcel Split (Zero-Leakage)", color="#d95f02", edgecolor="#8c3b00")

    ax2_a.set_ylabel("Overall Accuracy (OA %)", fontsize=11, fontweight="bold")
    ax2_a.set_title("(A) Architectural Accuracy Across Splitting Regimes", fontsize=12, fontweight="bold")
    ax2_a.set_xticks(x)
    ax2_a.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax2_a.set_ylim(0, 110)
    ax2_a.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2_a.legend(frameon=True, facecolor="#f8f9fa", loc="upper right")

    for rect in rects1:
        height = rect.get_height()
        ax2_a.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax2_a.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Generalization Cliff Delta (OA Drop)
    drops = [rnd - prc for rnd, prc in zip(oa_rnd, oa_prc)]
    bar_colors = ["#c0392b", "#e67e22", "#27ae60"]
    rects3 = ax2_b.bar(models, drops, width=0.5, color=bar_colors, edgecolor="#2c3e50")

    ax2_b.set_ylabel("Generalization Cliff Magnitude (Δ OA %)", fontsize=11, fontweight="bold")
    ax2_b.set_title("(B) Generalization Cliff Severity (Leakage Vulnerability)", fontsize=12, fontweight="bold")
    ax2_b.set_ylim(0, 80)
    ax2_b.grid(True, axis="y", linestyle="--", alpha=0.5)

    for rect in rects3:
        height = rect.get_height()
        ax2_b.annotate(f"-{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig2_path = os.path.join(FIG_DIR, "day05_transformer_vs_cnn_benchmark.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  -> Saved Figure 2: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Biophysical Attention Rollout & Wavelength Attribution
    # -------------------------------------------------------------------------
    fig3, (ax3_a, ax3_b) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Panel A: Spatial Attention Heatmap from [CLS]
    im_sa = ax3_a.imshow(attn_data["spatial_attn_map"], cmap="viridis", interpolation="gaussian")
    ax3_a.set_title("(A) SpectralFormer [CLS] Spatial Receptive Field Attention", fontsize=12, fontweight="bold")
    ax3_a.set_xlabel("Patch Width (Tokens)", fontsize=10)
    ax3_a.set_ylabel("Patch Height (Tokens)", fontsize=10)
    fig3.colorbar(im_sa, ax=ax3_a, fraction=0.046, pad=0.04)

    # Panel B: Component Importance Attribution
    comps = np.arange(1, 31)
    ax3_b.bar(comps, attn_data["component_importance"] * 100, color="#16a085", edgecolor="#0e6655")
    ax3_b.set_title("(B) Attention Projection Weight Across Spectral Eigenbands", fontsize=12, fontweight="bold")
    ax3_b.set_xlabel("PCA Spectral Eigenband Index (1–30)", fontsize=10)
    ax3_b.set_ylabel("Normalized Attention Contribution (%)", fontsize=10)
    ax3_b.set_xticks(np.arange(1, 31, 2))
    ax3_b.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig3_path = os.path.join(FIG_DIR, "day05_spectral_attention_rollout.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  -> Saved Figure 3: {fig3_path}")

    # -------------------------------------------------------------------------
    # Figure 4: Cross-Scene Domain Transfer (Salinas -> Indian Pines)
    # -------------------------------------------------------------------------
    fig4, (ax4_a, ax4_b) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # PCA projection of domain features
    pca_vis = PCA(n_components=2, random_state=42)
    feats_s = domain_results["feats_source"]
    feats_t = domain_results["feats_target"]
    combined = np.vstack([feats_s, feats_t])
    proj = pca_vis.fit_transform(combined)

    proj_s = proj[:len(feats_s)]
    proj_t = proj[len(feats_s):]

    ax4_a.scatter(proj_s[:, 0], proj_s[:, 1], color="#3498db", alpha=0.6, s=25, label="Source: Salinas Valley (CA)")
    ax4_a.scatter(proj_t[:, 0], proj_t[:, 1], color="#e74c3c", alpha=0.6, s=25, label="Target: Indian Pines (IN)")
    ax4_a.set_title("(A) Shared Latent Space Under CORAL Covariance Alignment", fontsize=12, fontweight="bold")
    ax4_a.set_xlabel("Latent Eigen-Coordinate 1", fontsize=10)
    ax4_a.set_ylabel("Latent Eigen-Coordinate 2", fontsize=10)
    ax4_a.grid(True, linestyle="--", alpha=0.5)
    ax4_a.legend(frameon=True, facecolor="#f8f9fa")

    # Panel B: Transfer Accuracy Comparison
    transfer_modes = ["Source Only (Zero-Shot)", "CORAL Aligned Transformer", "Target Supervised Oracle"]
    transfer_accs = [domain_results["zero_shot_oa"] * 100, domain_results["coral_oa"] * 100, benchmark_results["parcel_split"]["SpectralFormer"]["overall_accuracy_oa"] * 100]
    colors_trans = ["#7f8c8d", "#2980b9", "#27ae60"]

    rects_t = ax4_b.bar(transfer_modes, transfer_accs, width=0.45, color=colors_trans, edgecolor="#2c3e50")
    ax4_b.set_ylabel("Target Overall Accuracy (OA %)", fontsize=11, fontweight="bold")
    ax4_b.set_title("(B) Cross-Scene Generalization Benchmarking", fontsize=12, fontweight="bold")
    ax4_b.set_ylim(0, 100)
    ax4_b.grid(True, axis="y", linestyle="--", alpha=0.5)

    for rect in rects_t:
        height = rect.get_height()
        ax4_b.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig4_path = os.path.join(FIG_DIR, "day05_cross_scene_domain_transfer.png")
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    print(f"  -> Saved Figure 4: {fig4_path}")


# -----------------------------------------------------------------------------
# Master Execution Flow
# -----------------------------------------------------------------------------
def main():
    start_total_pipeline = time.time()
    print("=" * 80)
    print("TerraSpectra: Day 5 - Hyperspectral Vision Transformers & Self-Supervised MAE")
    print("=" * 80)

    # 1. Ingestion & Preprocessing
    data = load_and_prepare_hyperspectral_scenes()
    splits = construct_disjoint_parcel_splits(data["ip_gt"], data["coords"], data["labels"])

    # 2. Self-Supervised HS-MAE Pre-training
    mae_model, mae_history, mae_time = pretrain_hyperspectral_mae(
        data["X_mae_pretrain"], epochs=12, batch_size=128, lr=1e-3
    )

    # 3. Supervised Benchmark Suite
    # We benchmark HybridSN, SpectralFormer, and HS-MAE (Fine-Tuned) under both splits
    benchmark_results = {"random_split": {}, "parcel_split": {}}
    training_histories = {"random_split": {}, "parcel_split": {}}

    # Models instantiation factories
    models_to_test = [
        ("HybridSN", lambda: HybridSN(in_channels=1, num_classes=16)),
        ("SpectralFormer", lambda: SpectralFormer(in_channels=30, num_tokens=49, embed_dim=64, num_layers=3, num_heads=4, num_classes=16)),
        ("HS-MAE (Fine-Tuned)", lambda: MAEClassifier(mae_model, num_classes=16, freeze_encoder=False))
    ]

    print("\n[4/8] Running Architectural Benchmark Suite across Random & Parcel Splits...")
    trained_vit_model = None

    for model_name, model_fn in models_to_test:
        print(f"\n--- Benchmarking: {model_name} ---")
        # Regime A: Random Split
        m_rnd = model_fn()
        res_rnd, hist_rnd, _ = train_and_evaluate_classifier(
            m_rnd, model_name, data["X_ip_patches"], data["labels"], splits, "random",
            epochs=10, batch_size=128, lr=1e-3
        )
        benchmark_results["random_split"][model_name] = res_rnd
        training_histories["random_split"][model_name] = hist_rnd

        # Regime B: Parcel Split
        m_prc = model_fn()
        res_prc, hist_prc, trained_m = train_and_evaluate_classifier(
            m_prc, model_name, data["X_ip_patches"], data["labels"], splits, "parcel",
            epochs=10, batch_size=128, lr=1e-3
        )
        benchmark_results["parcel_split"][model_name] = res_prc
        training_histories["parcel_split"][model_name] = hist_prc

        if model_name == "SpectralFormer":
            trained_vit_model = trained_m

    # 4. Cross-Scene Domain Adaptation
    domain_results = run_cross_scene_domain_adaptation(
        data["X_sal_patches"], data["y_sal"], data["X_ip_patches"], data["labels"],
        epochs=10, batch_size=128, lr=1e-3
    )

    # 5. Attention Rollout Extraction
    attn_data = extract_spectral_attention_rollout(trained_vit_model, data["X_ip_patches"])

    # 6. Save Data Artifacts
    print("\n[5/8] Exporting benchmark artifacts and analytical tables...")
    json_path = os.path.join(PROC_DATA_DIR, "day05_transformer_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"  -> Exported: {json_path}")

    mae_hist_path = os.path.join(PROC_DATA_DIR, "day05_mae_pretraining_history.json")
    with open(mae_hist_path, "w") as f:
        json.dump(mae_history, f, indent=4)
    print(f"  -> Exported: {mae_hist_path}")

    hist_path = os.path.join(PROC_DATA_DIR, "day05_transformer_training_histories.json")
    with open(hist_path, "w") as f:
        json.dump(training_histories, f, indent=4)
    print(f"  -> Exported: {hist_path}")

    # Per-class summary CSV
    per_class_rows = []
    for regime in ["random_split", "parcel_split"]:
        for m_name in benchmark_results[regime]:
            pcm = benchmark_results[regime][m_name]["per_class_metrics"]
            for cls_name, vals in pcm.items():
                per_class_rows.append({
                    "regime": regime,
                    "model": m_name,
                    "class_name": cls_name,
                    "precision": vals["precision"],
                    "recall": vals["recall"],
                    "f1_score": vals["f1_score"]
                })
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_csv = os.path.join(PROC_DATA_DIR, "day05_per_class_transformer_metrics.csv")
    df_per_class.to_csv(per_class_csv, index=False)
    print(f"  -> Exported: {per_class_csv}")

    # Domain adaptation CSV
    df_domain = pd.DataFrame([{
        "source_dataset": "Salinas Valley (CA)",
        "target_dataset": "Indian Pines (IN)",
        "zero_shot_oa": domain_results["zero_shot_oa"],
        "coral_aligned_oa": domain_results["coral_oa"],
        "absolute_gain_pct": domain_results["gain_percent"]
    }])
    domain_csv = os.path.join(PROC_DATA_DIR, "day05_cross_scene_adaptation_results.csv")
    df_domain.to_csv(domain_csv, index=False)
    print(f"  -> Exported: {domain_csv}")

    # Spectral attention weights CSV
    df_attn = pd.DataFrame({
        "pca_component": np.arange(1, 31),
        "attention_weight": attn_data["component_importance"]
    })
    attn_csv = os.path.join(PROC_DATA_DIR, "day05_spectral_attention_weights.csv")
    df_attn.to_csv(attn_csv, index=False)
    print(f"  -> Exported: {attn_csv}")

    # 7. Render Figures
    generate_publication_figures(
        mae_history, mae_model, data["X_ip_patches"], benchmark_results, attn_data, domain_results
    )

    total_time = time.time() - start_total_pipeline
    print("=" * 80)
    print(f"TerraSpectra Day 5 Pipeline Successfully Executed in {total_time:.2f} seconds!")
    print("=" * 80)


if __name__ == "__main__":
    main()
