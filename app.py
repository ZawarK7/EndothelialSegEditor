# app.py
# Endothelial Cell Segmentation Editor (PyQt5 + Matplotlib canvas)
# - Drag PANNING + Zoom In/Out (buttons + mouse wheel zoom at cursor)
# - Device-aware METRICS (CellChek 20 / CellChek D/D+ / Concerto + optional Custom)
# - Saves metrics.json with explicit "units" map
# - Removed "Reset Zoom" BUTTON (reset logic still used internally on load/inference)
# - [NEW] Brightness / Contrast sliders
# - [NEW] Save Analysis Report: image + overlay + metrics in one figure

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QMessageBox,
    QGroupBox, QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
    QCheckBox, QSplitter, QComboBox, QDoubleSpinBox, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from skimage.segmentation import watershed
from skimage import measure
from skimage.morphology import binary_closing, disk, binary_dilation

# METRICS imports
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation as ndi_binary_dilation
from scipy.spatial import Voronoi


# =============================================================================
# PyInstaller-friendly resource helper
# =============================================================================
def resource_path(relative: str) -> str:
    """Return absolute path to resource, works for dev and PyInstaller onefile/onedir."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(Path(base) / relative)
    return str(Path(relative).resolve())


# =============================================================================
# CONFIGURATION (requested paths)
# =============================================================================
BASE_DIR = Path(resource_path("AI_based_solution"))
MODEL_PATH = Path(resource_path("checkpoints/best_model.pth"))

DESKTOP = Path.home() / "Desktop"
CORRECT_DIR = DESKTOP / "EndothelialSegEditor" / "approved_predictions"
EDITED_DIR  = DESKTOP / "EndothelialSegEditor" / "edited_predictions"

IMAGE_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_BOUNDARY_TH = 0.5
DEFAULT_CENTER_TH = 0.3
DEFAULT_MIN_AREA = 50

# Saved output mask mode: "dot" or "boundary" (used for CORRECT saves)
OUTPUT_MASK_MODE = "dot"
DOT_RADIUS_PX = 1
SAVE_BOUNDARY_THICKNESS = 2

# Display mode
DEFAULT_DISPLAY_MODE = "boundary"   # "boundary" or "dot"
DEFAULT_DISPLAY_THICKNESS = 3       # boundary line thickness on screen
DEFAULT_DISPLAY_DOT_RADIUS = 1      # dot radius on screen

# Editing defaults
DEFAULT_BRUSH_SIZE = 20             # lasso line thickness (preview)
DEFAULT_ERASE_RADIUS = 6            # adjustable: user can set 3..60 etc.

# ZOOM config
ZOOM_STEP_IN = 1.25
ZOOM_STEP_OUT = 1 / 1.25
MAX_ZOOM = 30.0


# =============================================================================
# DEVICE REGISTRY (3 required options + optional Custom)
# =============================================================================
DEVICE_REGISTRY = {
    "CellChek 20": {
        "width_um": 250.0,
        "height_um": 550.0,
        "area_mm2": 0.1375,
        "description": "Konan CellChek 20 (Full photographic field)"
    },
    "CellChek D/D+": {
        "width_um": 400.0,
        "height_um": 300.0,
        "area_mm2": 0.12,
        "description": "Konan CellChek D/D+ (Maximum analysis area)"
    },
    "Concerto Grader Charter": {
        "width_um": 240.0,
        "height_um": 400.0,
        "area_mm2": 0.096,
        "description": "According to Concerto Grader Charter (Maximum analysis area)"
    },
    "Custom": {
        "width_um": 280.0,
        "height_um": 320.0,
        "area_mm2": 0.0896,
        "description": "Custom (user-defined)"
    }
}
DEFAULT_DEVICE_NAME = "CellChek 20"


# =============================================================================
# HELPERS (masks + thumbnails)
# =============================================================================
def instance_to_boundary_mask(instance_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Convert instance labels to a BINARY boundary mask (0/255)."""
    if instance_mask is None:
        return None
    h, w = instance_mask.shape
    boundary = np.zeros((h, w), dtype=np.uint8)

    ids = np.unique(instance_mask)
    ids = ids[ids > 0]
    for cell_id in ids:
        m = (instance_mask == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cv2.drawContours(boundary, contours, -1, 255, max(1, int(thickness)))
    return boundary


def instance_to_dot_mask(instance_mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Convert instance labels to a BINARY dot mask (0/255), one dot per centroid."""
    if instance_mask is None:
        return None
    h, w = instance_mask.shape
    dots = np.zeros((h, w), dtype=np.uint8)

    for r in measure.regionprops(instance_mask):
        cy, cx = r.centroid
        x = int(np.clip(round(cx), 0, w - 1))
        y = int(np.clip(round(cy), 0, h - 1))
        cv2.circle(dots, (x, y), max(1, int(radius)), 255, -1)
    return dots


def bgr_to_qpixmap(bgr: np.ndarray, max_side: int = 180) -> QPixmap:
    """Create thumbnail QPixmap from a BGR image."""
    if bgr is None:
        return QPixmap()
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hh, ww = rgb.shape[:2]
    qimg = QImage(rgb.data, ww, hh, 3 * ww, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def deterministic_color_from_id(cell_id: int) -> tuple:
    """Deterministic BGR color per cell_id (stable across frames/edits)."""
    x = (cell_id * 2654435761) & 0xFFFFFFFF
    hue = x % 180
    sat = 200
    val = 255
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# =============================================================================
# [NEW] BRIGHTNESS / CONTRAST HELPER
# =============================================================================
def apply_brightness_contrast(img_bgr: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """
    Apply brightness and contrast adjustment to a BGR image.
    brightness: -100 to +100
    contrast:   -100 to +100
    """
    img = img_bgr.astype(np.float32)

    # Contrast: scale around mid-gray (128)
    if contrast != 0:
        factor = (259.0 * (contrast + 255)) / (255.0 * (259 - contrast))
        img = factor * (img - 128.0) + 128.0

    # Brightness: simple offset
    if brightness != 0:
        img = img + float(brightness)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# =============================================================================
# MODEL DEFINITION
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.boundary_head = nn.Conv2d(base_channels, 1, 1)
        self.distance_head = nn.Conv2d(base_channels, 1, 1)
        self.center_head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)

        boundary = torch.sigmoid(self.boundary_head(d1))
        distance = torch.sigmoid(self.distance_head(d1))
        center = torch.sigmoid(self.center_head(d1))
        return {'boundary': boundary, 'distance': distance, 'center': center}


# =============================================================================
# INFERENCE
# =============================================================================
def predict_image(model, img_path: Path, device, input_size=512):
    orig_img = cv2.imread(str(img_path))
    if orig_img is None:
        return None, None, None, None

    orig_h, orig_w = orig_img.shape[:2]
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    model.eval()
    with torch.no_grad():
        out = model(img_tensor)

    boundary = out['boundary'][0, 0].detach().cpu().numpy()
    distance = out['distance'][0, 0].detach().cpu().numpy()
    center = out['center'][0, 0].detach().cpu().numpy()

    boundary = cv2.resize(boundary, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    distance = cv2.resize(distance, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    center = cv2.resize(center, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return boundary, distance, center, orig_img


def extract_instances(boundary_pred, distance_pred, center_pred,
                      boundary_th=0.5, center_th=0.3, min_area=50):
    boundary_bin = (boundary_pred > boundary_th).astype(np.uint8)
    cell_mask = 1 - boundary_bin

    center_bin = (center_pred > center_th).astype(np.uint8)

    if center_bin.sum() == 0:
        if distance_pred.max() > 0 and cell_mask.sum() > 0:
            yy, xx = np.unravel_index(np.argmax(distance_pred * cell_mask), cell_mask.shape)
            center_bin[yy, xx] = 1

    _, markers_labeled = cv2.connectedComponents(center_bin.astype(np.uint8))

    if distance_pred.max() > 0:
        watershed_input = (distance_pred * 255).astype(np.uint8)
    else:
        watershed_input = (cell_mask * 255).astype(np.uint8)

    instance_mask = watershed(-watershed_input, markers_labeled, mask=cell_mask)

    for region_id in np.unique(instance_mask):
        if region_id == 0:
            continue
        region_area = int(np.sum(instance_mask == region_id))
        if region_area < int(min_area):
            instance_mask[instance_mask == region_id] = 0

    return measure.label(instance_mask, connectivity=2)


# =============================================================================
# METRICS (aligned with your reference + explicit units)
# =============================================================================
def compute_hexagonality_from_boundaries(instance_mask: np.ndarray):
    n_cells = int(instance_mask.max())
    if n_cells <= 0:
        return 0.0, []

    neighbor_counts = []
    for cell_id in range(1, n_cells + 1):
        cell_mask = (instance_mask == cell_id)
        if not np.any(cell_mask):
            neighbor_counts.append(0)
            continue

        dilated = ndi_binary_dilation(cell_mask, iterations=1)
        neighbors = np.unique(instance_mask[dilated])
        neighbors = neighbors[(neighbors != 0) & (neighbors != cell_id)]
        neighbor_counts.append(int(len(neighbors)))

    hex_count = sum(1 for n in neighbor_counts if n == 6)
    return (hex_count / float(n_cells)) * 100.0, neighbor_counts


def compute_hexagonality_from_voronoi(centers_yx: np.ndarray):
    if centers_yx is None or len(centers_yx) < 4:
        return 0.0, []

    points = centers_yx[:, [1, 0]]  # (x, y)
    try:
        vor = Voronoi(points)

        adj = [set() for _ in range(len(points))]
        for a, b in vor.ridge_points:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

        neighbor_counts = [len(adj[i]) for i in range(len(points))]
        hex_count = sum(1 for n in neighbor_counts if n == 6)
        return (hex_count / float(len(points))) * 100.0, neighbor_counts
    except Exception:
        return 0.0, [0] * len(centers_yx)


def calculate_endothelial_metrics_from_instance_mask(
    instance_mask: np.ndarray,
    device_cfg: dict,
    roi_width_px: int,
    roi_height_px: int,
):
    if instance_mask is None:
        return None

    props = regionprops(instance_mask)
    n_cells = len(props)
    if n_cells == 0:
        return None

    roi_width_px = int(roi_width_px)
    roi_height_px = int(roi_height_px)

    A_ROI_px2 = roi_width_px * roi_height_px
    A_G_px2 = int(np.sum(instance_mask > 0))

    cell_areas_px2 = np.array([p.area for p in props], dtype=np.float64)

    W_um = float(device_cfg["width_um"])
    H_um = float(device_cfg["height_um"])
    if "area_mm2" in device_cfg and device_cfg["area_mm2"] is not None:
        A_ROI_mm2 = float(device_cfg["area_mm2"])
    else:
        A_ROI_mm2 = (W_um * H_um) / 1e6

    um_per_px_x = W_um / float(roi_width_px)
    um_per_px_y = H_um / float(roi_height_px)
    A_px_um2 = um_per_px_x * um_per_px_y

    cell_areas_um2 = cell_areas_px2 * A_px_um2
    centers = np.array([p.centroid for p in props], dtype=np.float64)  # (y, x)

    HEX = 0.0
    neighbor_counts = []
    hex_method = "none"

    HEX_v, neigh_v = compute_hexagonality_from_voronoi(centers)
    if HEX_v > 0.0:
        HEX, neighbor_counts, hex_method = HEX_v, neigh_v, "voronoi"
    else:
        HEX_b, neigh_b = compute_hexagonality_from_boundaries(instance_mask)
        HEX, neighbor_counts, hex_method = HEX_b, neigh_b, "boundaries"

    if len(cell_areas_px2) > 1 and np.mean(cell_areas_px2) > 0:
        CV = (np.std(cell_areas_px2, ddof=1) / np.mean(cell_areas_px2)) * 100.0
    else:
        CV = 0.0

    AVE = float(np.mean(cell_areas_um2))
    MAX = float(np.max(cell_areas_um2))
    MIN = float(np.min(cell_areas_um2))
    SD = float(np.std(cell_areas_um2, ddof=1)) if len(cell_areas_um2) > 1 else 0.0

    NUM = int(n_cells)
    CD_A = NUM / A_ROI_mm2 if A_ROI_mm2 > 0 else 0.0

    A_G_mm2 = (A_G_px2 * A_px_um2) / 1e6
    CD_B = NUM / A_G_mm2 if A_G_mm2 > 0 else 0.0

    gradable_percentage = (A_G_px2 / float(A_ROI_px2)) * 100.0 if A_ROI_px2 > 0 else 0.0

    metrics = {
        "device": str(device_cfg.get("description", "Unknown")),
        "device_key": str(device_cfg.get("key", "")),

        "roi_width_um": W_um,
        "roi_height_um": H_um,
        "roi_area_mm2": float(A_ROI_mm2),

        "roi_width_px": int(roi_width_px),
        "roi_height_px": int(roi_height_px),
        "roi_area_px2": int(A_ROI_px2),

        "gradable_area_px2": int(A_G_px2),
        "gradable_percentage": float(gradable_percentage),

        "um_per_px_x": float(um_per_px_x),
        "um_per_px_y": float(um_per_px_y),
        "area_per_px_um2": float(A_px_um2),

        "NUM": int(NUM),
        "HEX": float(HEX),
        "HEX_method": str(hex_method),
        "CV": float(CV),

        "AVE": float(AVE),
        "MAX": float(MAX),
        "MIN": float(MIN),
        "SD": float(SD),

        "CD_A": float(CD_A),
        "gradable_area_mm2": float(A_G_mm2),
        "CD_B": float(CD_B),

        "neighbor_counts": neighbor_counts,
        "cell_areas_um2": cell_areas_um2.astype(np.float64).tolist(),

        "units": {
            "roi_width_um": "um",
            "roi_height_um": "um",
            "roi_area_mm2": "mm^2",
            "roi_width_px": "px",
            "roi_height_px": "px",
            "roi_area_px2": "px^2",
            "gradable_area_px2": "px^2",
            "gradable_percentage": "%",
            "um_per_px_x": "um/px",
            "um_per_px_y": "um/px",
            "area_per_px_um2": "um^2/px^2",
            "NUM": "count",
            "HEX": "%",
            "CV": "%",
            "AVE": "um^2",
            "MAX": "um^2",
            "MIN": "um^2",
            "SD": "um^2",
            "CD_A": "cells/mm^2",
            "gradable_area_mm2": "mm^2",
            "CD_B": "cells/mm^2",
        }
    }
    return metrics


# =============================================================================
# [NEW] ANALYSIS REPORT FIGURE
# =============================================================================
def build_analysis_report_figure(
    orig_img_bgr: np.ndarray,
    display_img_bgr: np.ndarray,
    instance_mask: np.ndarray,
    metrics: dict,
    image_name: str,
    brightness: int = 0,
    contrast: int = 0,
) -> plt.Figure:
    """
    Build and return a matplotlib Figure containing:
      Left col : Original image (with B/C applied) + Segmentation overlay
      Right col: Metrics summary panel (text) + Cell area histogram
    Layout (2 rows x 2 cols, plus title):
      [orig]   [overlay]
      [histogram]  [metrics text box]
    """
    fig = plt.figure(figsize=(18, 12), dpi=120)
    fig.patch.set_facecolor('#1a1a2e')

    gs = GridSpec(
        2, 2,
        figure=fig,
        left=0.04, right=0.97,
        top=0.90, bottom=0.06,
        wspace=0.12, hspace=0.30
    )

    ax_orig    = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_hist    = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])

    # ── Title ──────────────────────────────────────────────────────────────
    device_str = metrics.get("device", "Unknown Device") if metrics else "No metrics"
    fig.suptitle(
        f"ENDOTHELIAL ANALYSIS  ·  {image_name}\n{device_str}",
        fontsize=16, fontweight='bold',
        color='white', y=0.96
    )

    # ── Helper: show image ─────────────────────────────────────────────────
    def _show(ax, bgr, title):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=11, fontweight='bold', color='#e0e0e0', pad=6)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # Apply brightness/contrast to displayed original for the report
    orig_bc = apply_brightness_contrast(orig_img_bgr, brightness, contrast)
    _show(ax_orig, orig_bc, "Original Image")
    _show(ax_overlay, display_img_bgr, f"Segmentation Overlay  ({int(instance_mask.max())} cells)")

    # ── Histogram ──────────────────────────────────────────────────────────
    ax_hist.set_facecolor('#0f0f23')
    if metrics and metrics.get("cell_areas_um2"):
        areas = np.array(metrics["cell_areas_um2"])
        mean_a = float(np.mean(areas))
        ax_hist.hist(areas, bins=30, color='#4fc3f7', edgecolor='#1a1a2e', alpha=0.85)
        ax_hist.axvline(mean_a, color='#ff6b6b', linestyle='--', linewidth=1.8,
                        label=f'Mean: {mean_a:.1f}')
        ax_hist.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white',
                       edgecolor='#444466')
        ax_hist.set_xlabel('Cell Area (µm²)', color='#c0c0c0', fontsize=10)
        ax_hist.set_ylabel('Frequency', color='#c0c0c0', fontsize=10)
        ax_hist.set_title('Cell Area Distribution', fontsize=11,
                          fontweight='bold', color='#e0e0e0', pad=6)
        ax_hist.tick_params(colors='#c0c0c0')
        for spine in ax_hist.spines.values():
            spine.set_edgecolor('#444466')
    else:
        ax_hist.text(0.5, 0.5, 'No area data', ha='center', va='center',
                     color='#888', fontsize=12, transform=ax_hist.transAxes)
        ax_hist.set_title('Cell Area Distribution', fontsize=11,
                          fontweight='bold', color='#e0e0e0')

    # ── Metrics text panel ─────────────────────────────────────────────────
    ax_metrics.set_facecolor('#0f0f23')
    ax_metrics.axis('off')
    ax_metrics.set_title('Analysis Metrics', fontsize=11,
                          fontweight='bold', color='#e0e0e0', pad=6)

    if metrics:
        # Build neighbor distribution string
        nc = metrics.get("neighbor_counts", [])
        if nc:
            from collections import Counter
            cnt = Counter(nc)
            nd_str = "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        else:
            nd_str = "N/A"

        hex_method_label = metrics.get("HEX_method", "")
        if hex_method_label:
            hex_method_label = f" [{hex_method_label}]"

        lines = [
            ("DEVICE", ""),
            ("", metrics.get("device", "—")),
            ("", f"Analysis Area: {metrics.get('roi_width_um',0):.0f} × {metrics.get('roi_height_um',0):.0f} µm"
                 f"  ({metrics.get('roi_area_mm2',0):.4f} mm²)"),
            ("", f"Gradable Region: {metrics.get('gradable_percentage',0):.1f}%"
                 f"  ({metrics.get('gradable_area_mm2',0):.6f} mm²)"),
            ("", ""),
            ("SCALE-INDEPENDENT", ""),
            ("NUM", f"{metrics.get('NUM', '—')} cells"),
            ("HEX", f"{metrics.get('HEX', 0):.1f}%{hex_method_label}"),
            ("CV",  f"{metrics.get('CV', 0):.1f}%"),
            ("", ""),
            ("NEIGHBOR DISTRIBUTION", ""),
            ("", nd_str),
            ("", ""),
            ("CELL AREA (µm²)", ""),
            ("AVE", f"{metrics.get('AVE', 0):.1f}"),
            ("MAX", f"{metrics.get('MAX', 0):.1f}"),
            ("MIN", f"{metrics.get('MIN', 0):.1f}"),
            ("SD",  f"{metrics.get('SD', 0):.1f}"),
            ("", ""),
            ("CELL DENSITY (cells/mm²)", ""),
            ("CD_A (ROI)",      f"{metrics.get('CD_A', 0):.0f}"),
            ("CD_B (Gradable)", f"{metrics.get('CD_B', 0):.0f}"),
        ]

        y_pos = 0.97
        line_h = 0.043
        for key, val in lines:
            if key == "" and val == "":
                y_pos -= line_h * 0.6
                continue
            if val == "":
                # Section header
                ax_metrics.text(
                    0.03, y_pos, key,
                    transform=ax_metrics.transAxes,
                    fontsize=9, fontweight='bold',
                    color='#ffd54f', va='top'
                )
            elif key == "":
                # Continuation / sub-value
                ax_metrics.text(
                    0.06, y_pos, val,
                    transform=ax_metrics.transAxes,
                    fontsize=8.5, color='#b0bec5', va='top',
                    wrap=True
                )
            else:
                # Key : value pair
                ax_metrics.text(
                    0.06, y_pos, f"{key}:",
                    transform=ax_metrics.transAxes,
                    fontsize=9, fontweight='bold',
                    color='#80cbc4', va='top'
                )
                ax_metrics.text(
                    0.40, y_pos, val,
                    transform=ax_metrics.transAxes,
                    fontsize=9, color='#eceff1', va='top'
                )
            y_pos -= line_h

        # Decorative border around metrics panel
        rect = mpatches.FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.97,
            boxstyle="round,pad=0.01",
            linewidth=1.2,
            edgecolor='#444466',
            facecolor='none',
            transform=ax_metrics.transAxes,
            zorder=5
        )
        ax_metrics.add_patch(rect)

    else:
        ax_metrics.text(0.5, 0.5, 'No metrics available\n(Run inference first)',
                        ha='center', va='center', color='#888', fontsize=12,
                        transform=ax_metrics.transAxes)

    # Timestamp footer
    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    fig.text(0.99, 0.01, f"Generated: {ts}", ha='right', va='bottom',
             fontsize=7, color='#555577')

    return fig


# =============================================================================
# CANVAS
# =============================================================================
class InteractiveCanvas(FigureCanvas):
    edit_completed = pyqtSignal()

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(18, 10), dpi=100)
        self.fig.subplots_adjust(left=0.001, right=0.999, bottom=0.001, top=0.92)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        self.orig_img = None
        self.instance_mask = None
        self.display_img = None

        self.boundary_pred = None
        self.distance_pred = None

        self.boundary_th = DEFAULT_BOUNDARY_TH
        self.min_area = DEFAULT_MIN_AREA

        # [NEW] Brightness / Contrast
        self.brightness = 0
        self.contrast = 0

        self.edit_mode = False
        self.edit_tool = 'add_freehand_lasso'

        self.lasso_line_thickness = DEFAULT_BRUSH_SIZE
        self.erase_radius = DEFAULT_ERASE_RADIUS

        self.display_mode = DEFAULT_DISPLAY_MODE  # "boundary" or "dot"
        self.display_boundary_thickness = DEFAULT_DISPLAY_THICKNESS
        self.display_dot_radius = DEFAULT_DISPLAY_DOT_RADIUS
        self.multicolor_boundaries = True
        self.single_boundary_color_bgr = (0, 255, 0)

        self.drawing = False
        self.lasso_points = []
        self.preview_overlay = None

        self.last_erase_pos = None
        self.current_affected_ids = set()

        # -------- ZOOM / PAN state --------
        self._view_initialized = False
        self._full_xlim = None
        self._full_ylim = None
        self._max_zoom = float(MAX_ZOOM)

        self._panning = False
        self._pan_start_xy = None  # (x, y) in data coords
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('scroll_event', self.on_scroll)  # wheel zoom-at-cursor

        self.history = []
        self.history_index = -1

    # ---------- [NEW] brightness/contrast ----------
    def set_brightness(self, val: int):
        self.brightness = int(val)
        self.update_display()

    def set_contrast(self, val: int):
        self.contrast = int(val)
        self.update_display()

    def _get_display_base_img(self) -> np.ndarray:
        """Return orig_img with brightness/contrast applied."""
        if self.orig_img is None:
            return None
        return apply_brightness_contrast(self.orig_img, self.brightness, self.contrast)

    # ---------- zoom helpers ----------
    def _set_full_view(self):
        if self.orig_img is None:
            return
        h, w = self.orig_img.shape[:2]
        self._full_xlim = (-0.5, w - 0.5)
        self._full_ylim = (h - 0.5, -0.5)  # inverted y for image display
        self.ax.set_xlim(self._full_xlim)
        self.ax.set_ylim(self._full_ylim)
        self._view_initialized = True

    def reset_zoom(self):
        if self.orig_img is None:
            return
        self._set_full_view()
        self.draw_idle()

    def _current_zoom_ratio(self):
        if self.orig_img is None or self._full_xlim is None:
            return 1.0
        x0, x1 = self.ax.get_xlim()
        full_w = abs(self._full_xlim[1] - self._full_xlim[0])
        cur_w = abs(x1 - x0)
        if cur_w <= 1e-9:
            return self._max_zoom
        return float(full_w / cur_w)

    def _clamp_view_to_image(self, x0, x1, y0, y1):
        """Keep view within image bounds while preserving axis inversion."""
        if self.orig_img is None:
            return x0, x1, y0, y1

        h, w = self.orig_img.shape[:2]
        x_min, x_max = -0.5, w - 0.5
        y_min, y_max = -0.5, h - 0.5

        if x0 < x_min:
            shift = x_min - x0
            x0 += shift; x1 += shift
        if x1 > x_max:
            shift = x1 - x_max
            x0 -= shift; x1 -= shift

        y_low = min(y0, y1)
        y_high = max(y0, y1)
        if y_low < y_min:
            shift = y_min - y_low
            y0 += shift; y1 += shift
        if y_high > y_max:
            shift = y_high - y_max
            y0 -= shift; y1 -= shift

        return x0, x1, y0, y1

    def zoom(self, factor: float, center=None):
        if self.orig_img is None:
            return

        if self._full_xlim is None or self._full_ylim is None:
            self._set_full_view()

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        if center is None:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
        else:
            cx, cy = float(center[0]), float(center[1])

        cur_w = abs(x1 - x0)
        cur_h = abs(y1 - y0)

        new_w = cur_w / float(factor)
        new_h = cur_h / float(factor)

        full_w = abs(self._full_xlim[1] - self._full_xlim[0])
        full_h = abs(self._full_ylim[0] - self._full_ylim[1])

        if new_w > full_w:
            new_w = full_w
        if new_h > full_h:
            new_h = full_h

        min_w = full_w / self._max_zoom
        min_h = full_h / self._max_zoom
        if new_w < min_w:
            new_w = min_w
        if new_h < min_h:
            new_h = min_h

        nx0 = cx - new_w / 2.0
        nx1 = cx + new_w / 2.0
        ny0 = cy - new_h / 2.0
        ny1 = cy + new_h / 2.0

        nx0, nx1, ny0, ny1 = self._clamp_view_to_image(nx0, nx1, ny0, ny1)

        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny0, ny1)
        self._view_initialized = True
        self.draw_idle()

    def zoom_in(self):
        self.zoom(ZOOM_STEP_IN)

    def zoom_out(self):
        self.zoom(ZOOM_STEP_OUT)

    def on_scroll(self, event):
        if self.orig_img is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        step = getattr(event, "step", None)
        if step is None:
            if getattr(event, "button", "") == "up":
                step = 1
            elif getattr(event, "button", "") == "down":
                step = -1
            else:
                step = 0

        if step > 0:
            self.zoom(ZOOM_STEP_IN, center=(event.xdata, event.ydata))
        elif step < 0:
            self.zoom(ZOOM_STEP_OUT, center=(event.xdata, event.ydata))

    # ---------- pan helpers ----------
    def _pan_should_start(self, event):
        if event.inaxes != self.ax:
            return False
        if event.xdata is None or event.ydata is None:
            return False

        if event.button == 2:
            return True
        if not self.edit_mode and event.button == 1:
            return True
        if self.edit_mode and event.button == 3:
            return True
        return False

    def _start_pan(self, x, y):
        self._panning = True
        self._pan_start_xy = (float(x), float(y))
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()

        if self._full_xlim is None or self._full_ylim is None:
            self._set_full_view()

    def _do_pan(self, x, y):
        if not self._panning or self._pan_start_xy is None:
            return

        sx, sy = self._pan_start_xy
        dx = float(x) - sx
        dy = float(y) - sy

        x0, x1 = self._pan_start_xlim
        y0, y1 = self._pan_start_ylim

        nx0 = x0 - dx
        nx1 = x1 - dx
        ny0 = y0 - dy
        ny1 = y1 - dy

        nx0, nx1, ny0, ny1 = self._clamp_view_to_image(nx0, nx1, ny0, ny1)

        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny0, ny1)
        self._view_initialized = True
        self.draw_idle()

    def _stop_pan(self):
        self._panning = False
        self._pan_start_xy = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

    # ---------- public setters ----------
    def set_display_mode(self, mode: str):
        mode = str(mode).lower().strip()
        if mode in ("boundary", "dot"):
            self.display_mode = mode
            self.preview_overlay = None
            self.update_display()

    def set_display_boundary_thickness(self, t: int):
        self.display_boundary_thickness = int(max(1, t))
        self.update_display()

    def set_display_dot_radius(self, r: int):
        self.display_dot_radius = int(max(1, r))
        self.update_display()

    def set_multicolor_boundaries(self, enabled: bool):
        self.multicolor_boundaries = bool(enabled)
        self.update_display()

    def set_thresholds(self, boundary_th, min_area):
        self.boundary_th = float(boundary_th)
        self.min_area = int(min_area)

    def set_edit_tool(self, tool: str):
        self.edit_tool = tool
        self.lasso_points = []
        self.preview_overlay = None
        self.update_display()

    def set_lasso_line_thickness(self, t: int):
        self.lasso_line_thickness = int(max(1, t))

    def set_erase_radius(self, r: int):
        self.erase_radius = int(max(1, r))

    def toggle_edit_mode(self, enabled: bool):
        self.edit_mode = bool(enabled)
        self.drawing = False
        self.lasso_points = []
        self.preview_overlay = None
        self.last_erase_pos = None
        self.current_affected_ids.clear()
        self.update_display()

    # ---------- state mgmt ----------
    def clear_canvas(self):
        self.orig_img = None
        self.instance_mask = None
        self.display_img = None
        self.boundary_pred = None
        self.distance_pred = None
        self.preview_overlay = None
        self.lasso_points = []
        self.history = []
        self.history_index = -1
        self.last_erase_pos = None
        self.current_affected_ids.clear()

        self._view_initialized = False
        self._full_xlim = None
        self._full_ylim = None
        self._stop_pan()

        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_title("No image loaded", fontsize=13, fontweight='bold')
        self.draw()

    def set_image(self, orig_img, instance_mask, boundary_pred=None, distance_pred=None):
        self.orig_img = orig_img.copy()
        self.instance_mask = instance_mask.astype(np.int32).copy()

        self.boundary_pred = None if boundary_pred is None else boundary_pred.astype(np.float32).copy()
        self.distance_pred = None if distance_pred is None else distance_pred.astype(np.float32).copy()

        self._remove_wrapper_cells()

        self.history = [self.instance_mask.copy()]
        self.history_index = 0

        self.preview_overlay = None
        self.lasso_points = []

        self._view_initialized = False
        self._full_xlim = None
        self._full_ylim = None
        self._stop_pan()

        self.update_display()
        self.reset_zoom()

    # ---------- rendering ----------
    def _draw_boundaries(self, overlay: np.ndarray):
        ids = np.unique(self.instance_mask)
        ids = ids[ids > 0]

        thickness = int(max(1, self.display_boundary_thickness))

        for cell_id in ids:
            mask = (self.instance_mask == cell_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            if self.multicolor_boundaries:
                color = deterministic_color_from_id(int(cell_id))
            else:
                color = self.single_boundary_color_bgr

            cv2.drawContours(overlay, contours, -1, color, thickness)

    def _draw_dots(self, overlay: np.ndarray):
        rad = int(max(1, self.display_dot_radius))
        for r in measure.regionprops(self.instance_mask):
            cy, cx = r.centroid
            x = int(np.clip(round(cx), 0, overlay.shape[1] - 1))
            y = int(np.clip(round(cy), 0, overlay.shape[0] - 1))
            cv2.circle(overlay, (x, y), rad, (0, 255, 0), -1)

    def update_display(self):
        if self.orig_img is None:
            self.clear_canvas()
            return

        preserve_view = self._view_initialized
        prev_xlim = self.ax.get_xlim() if preserve_view else None
        prev_ylim = self.ax.get_ylim() if preserve_view else None

        # [MODIFIED] Use brightness/contrast-adjusted base image
        base_img = self._get_display_base_img()

        if self.instance_mask is None:
            self.ax.clear()
            self.ax.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
            self.ax.axis('off')
            self.ax.set_title("Raw image (no segmentation yet)", fontsize=13, fontweight='bold')
            if not self._view_initialized:
                self._set_full_view()
            else:
                self.ax.set_xlim(prev_xlim)
                self.ax.set_ylim(prev_ylim)
            self.draw()
            return

        overlay = base_img.copy()
        if self.display_mode == "boundary":
            self._draw_boundaries(overlay)
        else:
            self._draw_dots(overlay)

        if self.preview_overlay is not None:
            overlay = self.preview_overlay

        self.display_img = overlay

        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')

        n_cells = int(len(np.unique(self.instance_mask)) - 1)
        zoom_ratio = self._current_zoom_ratio()
        title = f"Cells: {n_cells} | View: {self.display_mode} | Zoom: {zoom_ratio:.2f}x"
        if self.edit_mode:
            title += f" | EDIT: {self.edit_tool}"
            title += " | Pan: RIGHT-drag (or middle)"
        else:
            title += " | Pan: LEFT-drag (or middle)"
        title += " | Wheel: zoom at cursor"
        self.ax.set_title(title, fontsize=13, fontweight='bold')

        if not self._view_initialized:
            self._set_full_view()
        else:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)

        self.draw()

    # ---------- history ----------
    def save_to_history(self):
        if self.instance_mask is None:
            return
        self.history = self.history[:self.history_index + 1]
        self.history.append(self.instance_mask.copy())
        self.history_index += 1
        if len(self.history) > 80:
            self.history.pop(0)
            self.history_index -= 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.instance_mask = self.history[self.history_index].copy()
            self.update_display()
            return True
        return False

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.instance_mask = self.history[self.history_index].copy()
            self.update_display()
            return True
        return False

    def get_current_mask(self):
        return None if self.instance_mask is None else self.instance_mask.copy()

    def _remove_wrapper_cells(self):
        if self.instance_mask is None:
            return False

        h, w = self.instance_mask.shape
        image_perimeter = 2 * (h + w)

        ids = np.unique(self.instance_mask)
        ids = ids[ids > 0]

        removed = False
        for cell_id in ids:
            mask = (self.instance_mask == cell_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            main_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(main_contour, True)

            if perimeter > image_perimeter * 0.8:
                self.instance_mask[self.instance_mask == cell_id] = 0
                removed = True
                continue

            x_coords = main_contour[:, 0, 0]
            y_coords = main_contour[:, 0, 1]
            touches_all_edges = (
                x_coords.min() <= 1 and
                x_coords.max() >= w - 2 and
                y_coords.min() <= 1 and
                y_coords.max() >= h - 2
            )
            if touches_all_edges:
                self.instance_mask[self.instance_mask == cell_id] = 0
                removed = True

        return removed

    # ---------- edit primitives ----------
    def remove_cell_at(self, pos):
        x, y = pos
        h, w = self.instance_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return False
        cell_id = int(self.instance_mask[y, x])
        if cell_id <= 0:
            return False
        self.instance_mask[self.instance_mask == cell_id] = 0
        return True

    def _relabel_if_split(self, affected_ids):
        if self.instance_mask is None:
            return

        max_id = int(self.instance_mask.max())

        for cid in affected_ids:
            if cid <= 0:
                continue
            mask = (self.instance_mask == cid).astype(np.uint8)
            if mask.sum() == 0:
                continue

            cc = measure.label(mask, connectivity=2)
            comps = np.unique(cc)
            comps = comps[comps > 0]

            if len(comps) <= 1:
                continue

            sizes = [(c, int((cc == c).sum())) for c in comps]
            sizes.sort(key=lambda t: t[1], reverse=True)

            keep_comp = sizes[0][0]
            new_mask = (cc == keep_comp)

            self.instance_mask[self.instance_mask == cid] = 0
            self.instance_mask[new_mask] = cid

            for comp, _sz in sizes[1:]:
                max_id += 1
                self.instance_mask[cc == comp] = max_id

    def erase_at(self, pos, collect_affected=False):
        x, y = pos
        h, w = self.instance_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return False

        r = int(max(1, self.erase_radius))

        x0 = max(0, x - r - 1)
        x1 = min(w, x + r + 2)
        y0 = max(0, y - r - 1)
        y1 = min(h, y + r + 2)

        region = self.instance_mask[y0:y1, x0:x1]
        if collect_affected:
            affected = set(np.unique(region).tolist())
            affected.discard(0)
            self.current_affected_ids.update(affected)

        cv2.circle(self.instance_mask, (x, y), r, 0, -1)
        return True

    # ---------- add tools ----------
    def commit_lasso_as_new_cell(self):
        if self.orig_img is None or self.instance_mask is None:
            return False
        if len(self.lasso_points) < 10:
            return False

        h, w = self.instance_mask.shape
        pts = np.array(self.lasso_points, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        poly = pts.reshape(-1, 1, 2)
        m = np.zeros((h, w), dtype=np.uint8)

        cv2.polylines(m, [poly], isClosed=True, color=255, thickness=max(2, self.lasso_line_thickness // 3))

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.fillPoly(m, contours, 255)

        close_r = max(1, self.lasso_line_thickness // 6)
        m = binary_closing(m.astype(bool), footprint=disk(close_r))

        m = m & (self.instance_mask == 0)

        if int(m.sum()) < int(self.min_area):
            return False

        labeled = measure.label(m, connectivity=2)
        regions = np.unique(labeled)
        regions = regions[regions > 0]

        if len(regions) == 0:
            return False

        next_id = int(self.instance_mask.max() + 1)
        added = False
        for region_id in regions:
            region_mask = (labeled == region_id)
            if int(region_mask.sum()) >= int(self.min_area):
                self.instance_mask[region_mask] = next_id
                next_id += 1
                added = True

        if added:
            self._remove_wrapper_cells()

        return added

    def fallback_circle_add(self, pos, restrict_mask=None):
        x, y = pos
        h, w = self.instance_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return False

        new_id = int(self.instance_mask.max() + 1)

        temp = np.zeros((h, w), np.uint8)
        cv2.circle(temp, (x, y), max(8, self.lasso_line_thickness), 255, -1)
        m = temp.astype(bool)

        if restrict_mask is not None:
            m = m & restrict_mask

        m = m & (self.instance_mask == 0)

        if int(m.sum()) < int(self.min_area):
            return False

        self.instance_mask[m] = new_id
        return True

    def auto_add_cell_from_center(self, pos):
        if self.boundary_pred is None or self.distance_pred is None:
            return self.fallback_circle_add(pos)

        x, y = pos
        h, w = self.instance_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return False

        allowed = (self.boundary_pred <= self.boundary_th)

        if not bool(allowed[y, x]):
            found = False
            for r in range(1, 16):
                yy0, yy1 = max(0, y - r), min(h, y + r + 1)
                xx0, xx1 = max(0, x - r), min(w, x + r + 1)
                sub = allowed[yy0:yy1, xx0:xx1]
                if np.any(sub):
                    ys, xs = np.where(sub)
                    y = int(yy0 + ys[0])
                    x = int(xx0 + xs[0])
                    found = True
                    break
            if not found:
                return self.fallback_circle_add(pos)

        d0 = float(self.distance_pred[y, x])
        if not np.isfinite(d0) or d0 <= 1e-6:
            d0 = float(np.nanmax(self.distance_pred))
            if not np.isfinite(d0) or d0 <= 1e-6:
                return self.fallback_circle_add((x, y), restrict_mask=allowed)

        k = 0.35
        seed_region = (self.distance_pred >= (k * d0)) & allowed

        lab = measure.label(seed_region.astype(np.uint8), connectivity=2)
        seed_lab = int(lab[y, x])
        if seed_lab == 0:
            return self.fallback_circle_add((x, y), restrict_mask=allowed)

        region = (lab == seed_lab)

        grown = region.copy()
        for _ in range(40):
            prev = grown
            grown = binary_dilation(grown, footprint=disk(2))
            grown = grown & allowed
            if np.array_equal(grown, prev):
                break

        write_mask = grown & (self.instance_mask == 0)

        if int(write_mask.sum()) < int(self.min_area):
            return False

        new_id = int(self.instance_mask.max() + 1)
        self.instance_mask[write_mask] = new_id
        return True

    # ---------- lasso preview ----------
    def update_lasso_preview(self):
        if self.orig_img is None or self.instance_mask is None:
            return

        if len(self.lasso_points) < 2:
            self.preview_overlay = None
            self.update_display()
            return

        base_img = self._get_display_base_img()
        ov = base_img.copy()
        if self.display_mode == "boundary":
            self._draw_boundaries(ov)
        else:
            self._draw_dots(ov)

        pts = np.array(self.lasso_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            ov, [pts], isClosed=False, color=(0, 255, 0),
            thickness=max(1, int(self.lasso_line_thickness // 2))
        )

        self.preview_overlay = ov
        self.update_display()

    # ---------- mouse handlers ----------
    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        if self._pan_should_start(event):
            self._start_pan(x, y)
            return

        if not self.edit_mode:
            return

        xi, yi = int(x), int(y)

        if self.edit_tool == 'add_freehand_lasso':
            self.drawing = True
            self.lasso_points = [(xi, yi)]
            self.preview_overlay = None
            self.update_lasso_preview()
            return

        if self.edit_tool == 'remove':
            changed = self.remove_cell_at((xi, yi))
            if changed:
                self.save_to_history()
                self.edit_completed.emit()
            self.update_display()
            return

        if self.edit_tool == 'erase':
            self.drawing = True
            self.current_affected_ids.clear()
            changed = self.erase_at((xi, yi), collect_affected=True)
            self.last_erase_pos = (xi, yi)
            if changed:
                self.update_display()
            return

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        if self._panning:
            self._do_pan(x, y)
            return

        if not self.drawing or not self.edit_mode:
            return

        xi, yi = int(x), int(y)

        if self.edit_tool == 'add_freehand_lasso':
            if len(self.lasso_points) == 0:
                self.lasso_points = [(xi, yi)]
            else:
                px, py = self.lasso_points[-1]
                if (xi - px) * (xi - px) + (yi - py) * (yi - py) >= 4:
                    self.lasso_points.append((xi, yi))
            self.update_lasso_preview()
            return

        if self.edit_tool == 'erase':
            if self.last_erase_pos is None:
                return
            prev_x, prev_y = self.last_erase_pos
            dx = xi - prev_x
            dy = yi - prev_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1:
                return
            num_steps = int(dist) + 1
            for i in range(1, num_steps + 1):
                ix = int(prev_x + (dx * i / num_steps))
                iy = int(prev_y + (dy * i / num_steps))
                self.erase_at((ix, iy), collect_affected=True)
            self.last_erase_pos = (xi, yi)
            self.update_display()
            return

    def on_release(self, event):
        if self._panning:
            self._stop_pan()
            return

        if not self.drawing:
            return
        self.drawing = False

        if self.edit_tool == 'add_freehand_lasso':
            changed = self.commit_lasso_as_new_cell()
            self.lasso_points = []
            self.preview_overlay = None
            if changed:
                self.save_to_history()
                self.edit_completed.emit()
            self.update_display()
            return

        if self.edit_tool == 'erase':
            if self.current_affected_ids:
                self._relabel_if_split(list(self.current_affected_ids))
            self.last_erase_pos = None
            self.current_affected_ids.clear()
            self.preview_overlay = None
            self.save_to_history()
            self.edit_completed.emit()
            self.update_display()
            return


# =============================================================================
# MAIN GUI
# =============================================================================
class SegmentationEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None
        self.current_images = []      # list[Path]
        self.current_index = -1
        self.current_image_path = None

        self.boundary_th = DEFAULT_BOUNDARY_TH
        self.center_th = DEFAULT_CENTER_TH
        self.min_area = DEFAULT_MIN_AREA

        self.last_tool = 'add_freehand_lasso'
        self.last_view = DEFAULT_DISPLAY_MODE

        self.inference_cache = {}  # {image_path: {'instance_mask', 'boundary_pred', 'distance_pred', 'orig_img'}}

        self.init_ui()
        self.load_model()
        self.canvas.clear_canvas()

    # ---------- UI ----------
    def init_ui(self):
        self.setWindowTitle('Endothelial Cell Segmentation Editor')
        self.setGeometry(50, 50, 2000, 1100)

        main = QWidget()
        self.setCentralWidget(main)
        root = QHBoxLayout(main)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel: thumbnails
        left = QWidget()
        left_layout = QVBoxLayout(left)

        left_title = QLabel("Loaded Images")
        left_title.setFont(QFont('Arial', 12, QFont.Bold))
        left_layout.addWidget(left_title)

        self.thumb_list = QListWidget()
        self.thumb_list.setIconSize(QSize(160, 160))
        self.thumb_list.itemSelectionChanged.connect(self.on_thumbnail_selected)
        left_layout.addWidget(self.thumb_list, stretch=1)

        left_btn_row = QHBoxLayout()
        btn_load_single = QPushButton("📁 Load Single")
        btn_load_single.clicked.connect(self.load_single_image)
        left_btn_row.addWidget(btn_load_single)

        btn_load_batch = QPushButton("📂 Load Batch")
        btn_load_batch.clicked.connect(self.load_batch_images)
        left_btn_row.addWidget(btn_load_batch)

        left_layout.addLayout(left_btn_row)
        splitter.addWidget(left)

        # Right panel: canvas + controls
        right = QWidget()
        right_layout = QVBoxLayout(right)

        title = QLabel('Interactive Endothelial Cell Segmentation Editor')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)

        self.canvas = InteractiveCanvas(self)
        self.canvas.edit_completed.connect(self.on_edit_completed)
        right_layout.addWidget(self.canvas, stretch=1)

        control_panel = self.create_control_panel()
        right_layout.addWidget(control_panel)

        self.status_label = QLabel('Ready. Load images to begin.')
        self.status_label.setStyleSheet('padding: 6px; background-color: #f0f0f0;')
        right_layout.addWidget(self.status_label)

        splitter.addWidget(right)
        splitter.setSizes([360, 1600])

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Row: inference/nav + zoom
        row1 = QHBoxLayout()

        self.run_inference_btn = QPushButton('🔮 Run Inference')
        self.run_inference_btn.setStyleSheet('background-color: #4CAF50; color: white; font-weight: bold;')
        self.run_inference_btn.clicked.connect(self.run_inference)
        row1.addWidget(self.run_inference_btn)

        prev_btn = QPushButton('⬅ Previous')
        prev_btn.clicked.connect(self.previous_image)
        row1.addWidget(prev_btn)

        self.image_counter = QLabel('Image 0 / 0')
        self.image_counter.setAlignment(Qt.AlignCenter)
        row1.addWidget(self.image_counter)

        next_btn = QPushButton('Next ➡')
        next_btn.clicked.connect(self.next_image)
        row1.addWidget(next_btn)

        zoom_in_btn = QPushButton('➕ Zoom In')
        zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        row1.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton('➖ Zoom Out')
        zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        row1.addWidget(zoom_out_btn)

        layout.addLayout(row1)

        # ── [NEW] Brightness / Contrast row ────────────────────────────────
        bc_group = QGroupBox("Image Adjustments")
        bc_layout = QHBoxLayout()

        bc_layout.addWidget(QLabel("☀ Brightness:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedWidth(160)
        self.brightness_slider.setTickInterval(25)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        bc_layout.addWidget(self.brightness_slider)

        self.brightness_label = QLabel("0")
        self.brightness_label.setFixedWidth(32)
        bc_layout.addWidget(self.brightness_label)

        bc_layout.addSpacing(20)

        bc_layout.addWidget(QLabel("◑ Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setFixedWidth(160)
        self.contrast_slider.setTickInterval(25)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        bc_layout.addWidget(self.contrast_slider)

        self.contrast_label = QLabel("0")
        self.contrast_label.setFixedWidth(32)
        bc_layout.addWidget(self.contrast_label)

        reset_bc_btn = QPushButton("↺ Reset")
        reset_bc_btn.setFixedWidth(60)
        reset_bc_btn.clicked.connect(self._reset_brightness_contrast)
        bc_layout.addWidget(reset_bc_btn)

        bc_layout.addStretch()
        bc_group.setLayout(bc_layout)
        layout.addWidget(bc_group)
        # ───────────────────────────────────────────────────────────────────

        # Device selection for metrics
        dev_group = QGroupBox("Device (metrics scaling)")
        dev_layout = QHBoxLayout()

        dev_layout.addWidget(QLabel("Select device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CellChek 20", "CellChek D/D+", "Concerto Grader Charter", "Custom"])
        if DEFAULT_DEVICE_NAME in ["CellChek 20", "CellChek D/D+", "Concerto Grader Charter", "Custom"]:
            self.device_combo.setCurrentText(DEFAULT_DEVICE_NAME)
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        dev_layout.addWidget(self.device_combo)

        dev_layout.addWidget(QLabel("Custom width (µm):"))
        self.custom_w = QDoubleSpinBox()
        self.custom_w.setRange(50.0, 2000.0)
        self.custom_w.setDecimals(1)
        self.custom_w.setSingleStep(10.0)
        self.custom_w.setValue(float(DEVICE_REGISTRY["Custom"]["width_um"]))
        self.custom_w.valueChanged.connect(self.on_custom_changed)
        dev_layout.addWidget(self.custom_w)

        dev_layout.addWidget(QLabel("Custom height (µm):"))
        self.custom_h = QDoubleSpinBox()
        self.custom_h.setRange(50.0, 2000.0)
        self.custom_h.setDecimals(1)
        self.custom_h.setSingleStep(10.0)
        self.custom_h.setValue(float(DEVICE_REGISTRY["Custom"]["height_um"]))
        self.custom_h.valueChanged.connect(self.on_custom_changed)
        dev_layout.addWidget(self.custom_h)

        self._sync_custom_enabled()
        dev_group.setLayout(dev_layout)
        layout.addWidget(dev_group)

        # Display options
        display_group = QGroupBox('Display Options')
        display_layout = QHBoxLayout()

        self.display_group_btns = QButtonGroup()

        self.view_boundary = QRadioButton('View: Boundaries')
        self.view_boundary.setChecked(DEFAULT_DISPLAY_MODE == "boundary")
        self.view_boundary.toggled.connect(lambda checked: self._set_view_if_checked("boundary", checked))
        self.display_group_btns.addButton(self.view_boundary)
        display_layout.addWidget(self.view_boundary)

        self.view_dots = QRadioButton('View: Center Dots')
        self.view_dots.setChecked(DEFAULT_DISPLAY_MODE == "dot")
        self.view_dots.toggled.connect(lambda checked: self._set_view_if_checked("dot", checked))
        self.display_group_btns.addButton(self.view_dots)
        display_layout.addWidget(self.view_dots)

        display_layout.addWidget(QLabel("Boundary thickness:"))
        self.boundary_thickness_spin = QSpinBox()
        self.boundary_thickness_spin.setRange(1, 12)
        self.boundary_thickness_spin.setValue(DEFAULT_DISPLAY_THICKNESS)
        self.boundary_thickness_spin.valueChanged.connect(self.canvas.set_display_boundary_thickness)
        display_layout.addWidget(self.boundary_thickness_spin)

        display_layout.addWidget(QLabel("Dot radius:"))
        self.dot_radius_spin = QSpinBox()
        self.dot_radius_spin.setRange(1, 8)
        self.dot_radius_spin.setValue(DEFAULT_DISPLAY_DOT_RADIUS)
        self.dot_radius_spin.valueChanged.connect(self.canvas.set_display_dot_radius)
        display_layout.addWidget(self.dot_radius_spin)

        self.cb_multicolor = QCheckBox("Multi-color boundaries")
        self.cb_multicolor.setChecked(True)
        self.cb_multicolor.stateChanged.connect(lambda s: self.canvas.set_multicolor_boundaries(s == Qt.Checked))
        display_layout.addWidget(self.cb_multicolor)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Edit tools
        edit_group = QGroupBox('Edit Tools')
        edit_layout = QHBoxLayout()

        self.edit_mode_btn = QPushButton('✏ Enter Edit Mode')
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.clicked.connect(self.toggle_edit_mode)
        edit_layout.addWidget(self.edit_mode_btn)

        self.tool_group = QButtonGroup()

        self.add_freehand_radio = QRadioButton('Add (Freehand Lasso)')
        self.add_freehand_radio.setChecked(True)
        self.add_freehand_radio.toggled.connect(lambda checked: self._set_tool_if_checked('add_freehand_lasso', checked))
        self.tool_group.addButton(self.add_freehand_radio)
        edit_layout.addWidget(self.add_freehand_radio)

        self.remove_radio = QRadioButton('Remove Cell')
        self.remove_radio.toggled.connect(lambda checked: self._set_tool_if_checked('remove', checked))
        self.tool_group.addButton(self.remove_radio)
        edit_layout.addWidget(self.remove_radio)

        self.erase_radio = QRadioButton('Erase')
        self.erase_radio.toggled.connect(lambda checked: self._set_tool_if_checked('erase', checked))
        self.tool_group.addButton(self.erase_radio)
        edit_layout.addWidget(self.erase_radio)

        edit_layout.addWidget(QLabel('Lasso thickness:'))
        self.lasso_spin = QSpinBox()
        self.lasso_spin.setRange(3, 120)
        self.lasso_spin.setValue(DEFAULT_BRUSH_SIZE)
        self.lasso_spin.valueChanged.connect(self.canvas.set_lasso_line_thickness)
        edit_layout.addWidget(self.lasso_spin)

        edit_layout.addWidget(QLabel('Erase size:'))
        self.erase_spin = QSpinBox()
        self.erase_spin.setRange(3, 60)
        self.erase_spin.setValue(DEFAULT_ERASE_RADIUS)
        self.erase_spin.valueChanged.connect(self.canvas.set_erase_radius)
        edit_layout.addWidget(self.erase_spin)

        undo_btn = QPushButton('↶ Undo')
        undo_btn.clicked.connect(self.canvas.undo)
        edit_layout.addWidget(undo_btn)

        redo_btn = QPushButton('↷ Redo')
        redo_btn.clicked.connect(self.canvas.redo)
        edit_layout.addWidget(redo_btn)

        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)

        # Save row
        save_row = QHBoxLayout()

        correct_btn = QPushButton('✓ Mark as Correct & Save')
        correct_btn.setStyleSheet('background-color: #2196F3; color: white; font-weight: bold; padding: 10px;')
        correct_btn.clicked.connect(self.save_as_correct)
        save_row.addWidget(correct_btn)

        save_edited_btn = QPushButton('💾 Save Edited Version')
        save_edited_btn.setStyleSheet('background-color: #FF9800; color: white; font-weight: bold; padding: 10px;')
        save_edited_btn.clicked.connect(self.save_as_edited)
        save_row.addWidget(save_edited_btn)

        layout.addLayout(save_row)

        return panel

    # ---------- [NEW] brightness/contrast handlers ----------
    def _on_brightness_changed(self, val: int):
        self.brightness_label.setText(str(val))
        self.canvas.set_brightness(val)

    def _on_contrast_changed(self, val: int):
        self.contrast_label.setText(str(val))
        self.canvas.set_contrast(val)

    def _reset_brightness_contrast(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)

    # ---------- device handlers ----------
    def _sync_custom_enabled(self):
        is_custom = (self.device_combo.currentText() == "Custom")
        self.custom_w.setEnabled(is_custom)
        self.custom_h.setEnabled(is_custom)

    def on_device_changed(self, _text):
        self._sync_custom_enabled()
        self.update_status(f"Device set to: {self.device_combo.currentText()}")

    def on_custom_changed(self, _val):
        if self.device_combo.currentText() == "Custom":
            self.update_status(f"Custom device: {self.custom_w.value():.1f} × {self.custom_h.value():.1f} µm")

    def _get_selected_device_cfg(self):
        name = self.device_combo.currentText()
        cfg = dict(DEVICE_REGISTRY[name])
        cfg["key"] = name

        if name == "Custom":
            w = float(self.custom_w.value())
            h = float(self.custom_h.value())
            cfg["width_um"] = w
            cfg["height_um"] = h
            cfg["area_mm2"] = (w * h) / 1e6
            cfg["description"] = f"Custom ({w:.0f} × {h:.0f} µm)"
        return cfg

    # ---------- callbacks ----------
    def update_status(self, message: str):
        self.status_label.setText(message)

    def _set_view_if_checked(self, mode, checked):
        if checked:
            self.last_view = mode
            self.canvas.set_display_mode(mode)
            self.update_status(f"View: {mode}")

    def _set_tool_if_checked(self, tool, checked):
        if checked:
            self.last_tool = tool
            self.canvas.set_edit_tool(tool)
            self.update_status(f"Tool: {tool}")

    def toggle_edit_mode(self):
        is_edit = self.edit_mode_btn.isChecked()
        self.canvas.toggle_edit_mode(is_edit)

        if is_edit:
            self.edit_mode_btn.setText('✓ Exit Edit Mode')
            self.edit_mode_btn.setStyleSheet('background-color: #f44336; color: white;')
            self.update_status("Edit mode ON (Pan with RIGHT-drag)")
        else:
            self.edit_mode_btn.setText('✏ Enter Edit Mode')
            self.edit_mode_btn.setStyleSheet('')
            self.update_status("Edit mode OFF (Pan with LEFT-drag)")

    def on_edit_completed(self):
        m = self.canvas.get_current_mask()
        if m is None:
            return
        num_cells = int(len(np.unique(m)) - 1)
        self.update_status(f'Edit applied. Current cells: {num_cells}')

        if self.current_image_path is not None:
            cache_key = str(self.current_image_path)
            if cache_key in self.inference_cache:
                self.inference_cache[cache_key]['instance_mask'] = m.copy()

    # ---------- model ----------
    def load_model(self):
        if not MODEL_PATH.exists():
            QMessageBox.critical(self, 'Error', f'Model not found at:\n{MODEL_PATH}')
            return

        try:
            self.model = MultiTaskUNet().to(DEVICE)
            checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            self.update_status(f'Model loaded: {MODEL_PATH.name}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load model:\n{str(e)}')

    # ---------- loading images + thumbnails ----------
    def _force_exit_edit_mode(self):
        if self.edit_mode_btn.isChecked():
            self.edit_mode_btn.blockSignals(True)
            self.edit_mode_btn.setChecked(False)
            self.edit_mode_btn.blockSignals(False)
        self.canvas.toggle_edit_mode(False)
        self.edit_mode_btn.setText('✏ Enter Edit Mode')
        self.edit_mode_btn.setStyleSheet('')

    def load_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '',
            'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)'
        )
        if not file_path:
            return

        self._force_exit_edit_mode()
        self.current_images = [Path(file_path)]
        self.current_index = 0
        self.current_image_path = None
        self.inference_cache.clear()

        self.populate_thumbnails()
        self.select_thumbnail(0)

        self.update_image_counter()
        self.update_status(f'Loaded 1 image: {Path(file_path).name}')

    def load_batch_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select Images', '',
            'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)'
        )
        if not file_paths:
            return

        self._force_exit_edit_mode()
        self.current_images = [Path(p) for p in file_paths]
        self.current_index = 0
        self.current_image_path = None
        self.inference_cache.clear()

        self.populate_thumbnails()
        self.select_thumbnail(0)

        self.update_image_counter()
        self.update_status(f'Loaded {len(file_paths)} images')

    def populate_thumbnails(self):
        self.thumb_list.clear()
        for p in self.current_images:
            bgr = cv2.imread(str(p))
            pix = bgr_to_qpixmap(bgr, max_side=160)
            item = QListWidgetItem(p.name)
            item.setIcon(QIcon(pix))
            item.setToolTip(str(p))
            self.thumb_list.addItem(item)

    def select_thumbnail(self, index: int):
        if 0 <= index < self.thumb_list.count():
            self.thumb_list.setCurrentRow(index)
            self.on_thumbnail_selected()

    def on_thumbnail_selected(self):
        row = self.thumb_list.currentRow()
        if row < 0 or row >= len(self.current_images):
            return

        self.current_index = row
        self.current_image_path = self.current_images[self.current_index]
        cache_key = str(self.current_image_path)

        if cache_key in self.inference_cache:
            cached = self.inference_cache[cache_key]
            self.canvas.set_thresholds(self.boundary_th, self.min_area)
            self.canvas.set_image(
                cached['orig_img'],
                cached['instance_mask'],
                cached['boundary_pred'],
                cached['distance_pred']
            )
            num_cells = int(len(np.unique(cached['instance_mask'])) - 1)
            self.update_status(f"Loaded cached segmentation: {self.current_image_path.name} ({num_cells} cells)")
        else:
            bgr = cv2.imread(str(self.current_image_path))
            if bgr is None:
                self.canvas.clear_canvas()
                self.update_status('Failed to load image')
                return

            self.canvas.orig_img = bgr.copy()
            self.canvas.instance_mask = None
            self.canvas.boundary_pred = None
            self.canvas.distance_pred = None
            self.canvas.display_img = bgr.copy()
            self.canvas.preview_overlay = None
            self.canvas.history = []
            self.canvas.history_index = -1
            self.canvas.lasso_points = []
            self.canvas.drawing = False
            self.canvas.edit_mode = False

            self.canvas._view_initialized = False
            self.canvas._full_xlim = None
            self.canvas._full_ylim = None
            self.canvas._stop_pan()

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.canvas.ax.clear()
            self.canvas.ax.imshow(rgb)
            self.canvas.ax.axis('off')
            self.canvas.ax.set_title(
                f"Raw Image: {self.current_image_path.name}\nClick 'Run Inference' to segment",
                fontsize=13, fontweight='bold'
            )
            self.canvas._set_full_view()
            self.canvas.fig.canvas.draw_idle()
            self.canvas.fig.canvas.flush_events()

            self.update_status(f'Loaded: {self.current_image_path.name} - Click "Run Inference" to segment')

        self.update_image_counter()

        self.canvas.display_mode = self.last_view
        self.canvas.edit_tool = self.last_tool

        if cache_key in self.inference_cache:
            self.canvas.update_display()

    # ---------- inference ----------
    def run_inference(self):
        if not self.current_images or self.current_index < 0:
            QMessageBox.warning(self, 'Warning', 'Please load images first')
            return
        if self.model is None:
            QMessageBox.critical(self, 'Error', 'Model not loaded')
            return

        self._force_exit_edit_mode()

        self.current_image_path = self.current_images[self.current_index]
        self.update_status(f'Running inference on {self.current_image_path.name}...')

        try:
            boundary, distance, center, orig_img = predict_image(
                self.model, self.current_image_path, DEVICE, IMAGE_SIZE
            )
            if boundary is None:
                QMessageBox.critical(self, 'Error', 'Failed to process image')
                return

            instance_mask = extract_instances(
                boundary, distance, center,
                self.boundary_th, self.center_th, self.min_area
            )

            cache_key = str(self.current_image_path)
            self.inference_cache[cache_key] = {
                'orig_img': orig_img.copy(),
                'instance_mask': instance_mask.copy(),
                'boundary_pred': boundary.copy(),
                'distance_pred': distance.copy()
            }

            self.canvas.set_thresholds(self.boundary_th, self.min_area)
            self.canvas.set_image(orig_img, instance_mask, boundary_pred=boundary, distance_pred=distance)
            self.canvas.reset_zoom()

            num_cells = int(len(np.unique(instance_mask)) - 1)
            self.update_status(f'Inference complete. Detected {num_cells} cells. (Pan/zoom enabled)')

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Inference failed:\n{str(e)}')

    # ---------- navigation ----------
    def previous_image(self):
        if self.current_images and self.current_index > 0:
            self.select_thumbnail(self.current_index - 1)

    def next_image(self):
        if self.current_images and self.current_index < len(self.current_images) - 1:
            self.select_thumbnail(self.current_index + 1)

    def update_image_counter(self):
        if self.current_images and self.current_index >= 0:
            self.image_counter.setText(f'Image {self.current_index + 1} / {len(self.current_images)}')
        else:
            self.image_counter.setText('Image 0 / 0')

    # ---------- [NEW] save_analysis_report ----------
    def save_analysis_report(self):
        """Save a combined analysis report PNG: original + overlay + metrics."""
        if self.current_image_path is None:
            QMessageBox.warning(self, 'Warning', 'No image loaded.')
            return

        instance_mask = self.canvas.get_current_mask()
        if instance_mask is None or int(np.unique(instance_mask).max()) == 0:
            QMessageBox.warning(self, 'Warning',
                                'No segmentation available.\nRun inference first.')
            return

        orig_img = self.canvas.orig_img
        display_img = self.canvas.display_img
        if display_img is None:
            display_img = orig_img

        dev_cfg = self._get_selected_device_cfg()
        roi_h_px, roi_w_px = orig_img.shape[:2]

        metrics = calculate_endothelial_metrics_from_instance_mask(
            instance_mask=instance_mask,
            device_cfg=dev_cfg,
            roi_width_px=roi_w_px,
            roi_height_px=roi_h_px,
        )

        fig = build_analysis_report_figure(
            orig_img_bgr=orig_img,
            display_img_bgr=display_img,
            instance_mask=instance_mask,
            metrics=metrics,
            image_name=self.current_image_path.name,
            brightness=self.canvas.brightness,
            contrast=self.canvas.contrast,
        )

        # Default save path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = self.current_image_path.stem
        default_dir = DESKTOP / "EndothelialSegEditor" / "analysis_reports"
        default_dir.mkdir(parents=True, exist_ok=True)
        default_path = str(default_dir / f"{base_name}_analysis_{timestamp}.png")

        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Analysis Report', default_path,
            'PNG Image (*.png);;PDF (*.pdf)'
        )

        if not save_path:
            plt.close(fig)
            return

        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            self.update_status(f'Analysis report saved: {Path(save_path).name}')
            QMessageBox.information(self, 'Saved',
                                    f'Analysis report saved to:\n{save_path}')
        except Exception as e:
            plt.close(fig)
            QMessageBox.critical(self, 'Error', f'Failed to save report:\n{str(e)}')

    # ---------- saving ----------
    def save_as_correct(self):
        if self.current_image_path is None:
            QMessageBox.warning(self, 'Warning', 'No image loaded')
            return
        self.save_result(CORRECT_DIR, 'approved')

    def save_as_edited(self):
        if self.current_image_path is None:
            QMessageBox.warning(self, 'Warning', 'No image loaded')
            return
        self.save_result(EDITED_DIR, 'edited')

    def save_result(self, output_dir: Path, label: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        masks_dir = output_dir / "masks"
        overlays_dir = output_dir / "overlays"
        originals_dir = output_dir / "originals"
        labels_dir = output_dir / "instance_labels_npy"
        metadata_dir = output_dir / "metrics"
        for d in [masks_dir, overlays_dir, originals_dir, labels_dir, metadata_dir]:
            d.mkdir(exist_ok=True, parents=True)

        dots_dir = masks_dir / "dots"
        boundaries_dir = masks_dir / "boundaries"
        dots_dir.mkdir(exist_ok=True, parents=True)
        boundaries_dir.mkdir(exist_ok=True, parents=True)

        instance_mask = self.canvas.get_current_mask()
        if instance_mask is None:
            QMessageBox.warning(self, 'Warning', 'Nothing to save (no mask).')
            return

        if int(np.unique(instance_mask).max()) == 0:
            QMessageBox.warning(self, 'Warning', 'No cells detected/edited to save. Run inference first.')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = self.current_image_path.stem

        saved_mask_paths = {}

        if label == "edited":
            dot_mask = instance_to_dot_mask(instance_mask, radius=DOT_RADIUS_PX)
            dot_path = dots_dir / f'{base_name}_{label}_dotMask_r{DOT_RADIUS_PX}_{timestamp}.png'
            cv2.imwrite(str(dot_path), dot_mask)
            saved_mask_paths["dot_mask"] = str(dot_path)

            boundary_mask = instance_to_boundary_mask(instance_mask, thickness=SAVE_BOUNDARY_THICKNESS)
            bnd_path = boundaries_dir / f'{base_name}_{label}_boundaryMask_t{SAVE_BOUNDARY_THICKNESS}_{timestamp}.png'
            cv2.imwrite(str(bnd_path), boundary_mask)
            saved_mask_paths["boundary_mask"] = str(bnd_path)

            saved_mask_paths["mask"] = str(dot_path)
        else:
            if OUTPUT_MASK_MODE.lower() == "dot":
                out_mask = instance_to_dot_mask(instance_mask, radius=DOT_RADIUS_PX)
                mask_path = dots_dir / f'{base_name}_{label}_dotMask_r{DOT_RADIUS_PX}_{timestamp}.png'
            elif OUTPUT_MASK_MODE.lower() == "boundary":
                out_mask = instance_to_boundary_mask(instance_mask, thickness=SAVE_BOUNDARY_THICKNESS)
                mask_path = boundaries_dir / f'{base_name}_{label}_boundaryMask_t{SAVE_BOUNDARY_THICKNESS}_{timestamp}.png'
            else:
                raise ValueError(f"Unknown OUTPUT_MASK_MODE: {OUTPUT_MASK_MODE}")

            cv2.imwrite(str(mask_path), out_mask)
            saved_mask_paths["mask"] = str(mask_path)

        overlay_path = overlays_dir / f'{base_name}_{label}_overlay_{timestamp}.png'
        cv2.imwrite(str(overlay_path), self.canvas.display_img)

        orig_path = originals_dir / f'{base_name}_{label}_original_{timestamp}.png'
        cv2.imwrite(str(orig_path), self.canvas.orig_img)

        labels_npy_path = labels_dir / f'{base_name}_{label}_instanceLabels_{timestamp}.npy'
        np.save(str(labels_npy_path), instance_mask.astype(np.int32))

        dev_cfg = self._get_selected_device_cfg()

        roi_h_px, roi_w_px = self.canvas.orig_img.shape[:2]
        metrics = calculate_endothelial_metrics_from_instance_mask(
            instance_mask=instance_mask,
            device_cfg=dev_cfg,
            roi_width_px=roi_w_px,
            roi_height_px=roi_h_px,
        )

        metrics_path = None
        if metrics is not None:
            metrics_path = metadata_dir / f'{base_name}_{label}_metrics_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        # Auto-save analysis report figure alongside other outputs
        reports_dir = output_dir / "analysis_reports"
        reports_dir.mkdir(exist_ok=True, parents=True)
        report_path = reports_dir / f'{base_name}_{label}_analysis_{timestamp}.png'
        try:
            fig = build_analysis_report_figure(
                orig_img_bgr=self.canvas.orig_img,
                display_img_bgr=self.canvas.display_img if self.canvas.display_img is not None else self.canvas.orig_img,
                instance_mask=instance_mask,
                metrics=metrics,
                image_name=self.current_image_path.name,
                brightness=self.canvas.brightness,
                contrast=self.canvas.contrast,
            )
            fig.savefig(str(report_path), dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)
        except Exception as e:
            report_path = None
            print(f"[Warning] Could not save analysis report: {e}")

        metadata = {
            'original_image': str(self.current_image_path),
            'timestamp': timestamp,
            'label': label,
            'num_cells': int(len(np.unique(instance_mask)) - 1),
            'boundary_threshold': float(self.boundary_th),
            'center_threshold': float(self.center_th),
            'min_area': int(self.min_area),
            'output_mask_mode': str(OUTPUT_MASK_MODE),
            'dot_radius_px': int(DOT_RADIUS_PX),
            'boundary_thickness_px': int(SAVE_BOUNDARY_THICKNESS),
            'display_mode_at_save': str(self.canvas.display_mode),
            'display_boundary_thickness': int(self.canvas.display_boundary_thickness),
            'multicolor_boundaries': bool(self.canvas.multicolor_boundaries),
            'edited': label == 'edited',
            'brightness': int(self.canvas.brightness),
            'contrast': int(self.canvas.contrast),
            'metrics': metrics,
            'saved_paths': {
                **saved_mask_paths,
                'overlay': str(overlay_path),
                'original': str(orig_path),
                'instance_labels_npy': str(labels_npy_path),
                'metrics_json': str(metrics_path) if metrics_path is not None else None,
                'analysis_report': str(report_path) if report_path else None,
            }
        }

        QMessageBox.information(
            self,
            'Success',
            f"Saved ({label})\n\n"
            f"Outputs written to:\n{output_dir}\n\n"
            f"Masks:\n  {dots_dir}\n  {boundaries_dir}\n\n"
            f"Device used for metrics:\n  {dev_cfg.get('description','')}\n"
            f"Metrics:\n  {metrics_path if metrics_path else 'None'}\n"
            f"Analysis Report:\n  {report_path if report_path else 'Not saved'}"
        )
        self.update_status(f'Saved as {label}: {output_dir}')


# =============================================================================
# MAIN
# =============================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    CORRECT_DIR.mkdir(exist_ok=True, parents=True)
    EDITED_DIR.mkdir(exist_ok=True, parents=True)

    window = SegmentationEditor()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()