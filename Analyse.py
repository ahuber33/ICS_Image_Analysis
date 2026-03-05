# -*- coding: utf-8 -*-
"""
Main script — GPU-enabled batch processing.
Requires: image_analysis_optimized.py (ICS_Include)
"""

import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import logging

import Include

logging.getLogger("cellpose").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*no mask pixels found.*")

# ═══════════════════════════════════════════════════════════════════════════
# 1. DEVICE  (auto-détection GPU / CPU)
# ═══════════════════════════════════════════════════════════════════════════

device = Include.get_device()   # affiche "✅ GPU" ou "⚠️ CPU"

# ═══════════════════════════════════════════════════════════════════════════
# 2. PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════
dossier = Path(r'D:\ICS\DM5000 63X Mixtes n=2\20251031- Mixtes DIV 15\Ctrl\Ctrl1_TubulinA594 + pTau488')

# Prétraitement
pretraitement_sigma = 1.5
clip_limit          = 0.02
distance_transform  = 40
margin              = 40

# CNN
cnn_threshold       = 0.23
patch_size          = 64
half                = patch_size // 2
cnn_batch_size      = 512   # ↑ augmenter si VRAM ≥ 6 Go, ↓ réduire si OOM

# Fusion patches
overlap_thresh      = 0.9

# Cellpose
diameter            = 8
cellprob_threshold  = 0.1
flow_threshold      = 0.5
min_area            = 5
max_area            = 500
min_circularity     = 0.5
max_axis_ratio      = 2
top_hat_radius      = 10
cellpose_batch_size = 32    # ↑ augmenter si VRAM ≥ 6 Go, ↓ réduire si OOM

# ═══════════════════════════════════════════════════════════════════════════
# 3. CHARGEMENT DES MODÈLES  (une seule fois, placés sur GPU)
# ═══════════════════════════════════════════════════════════════════════════

# ── CNN soma ────────────────────────────────────────────────────────────────
model_cnn = Include.SomaCNN()
model_cnn.load_state_dict(torch.load("soma_cnn_test.pth", map_location=device))
model_cnn = model_cnn.to(device)
model_cnn.eval()

# ── Cellpose cyto3  (gpu=True si CUDA disponible) ──────────────────────────
from cellpose import models as cp_models
cellpose_model = cp_models.Cellpose(
    gpu=torch.cuda.is_available(),
    model_type='cyto3'
)

# ═══════════════════════════════════════════════════════════════════════════
# 4. BATCH LOOP
# ═══════════════════════════════════════════════════════════════════════════
output_folder = dossier / "output"
output_folder.mkdir(exist_ok=True)

fichiers = list(dossier.glob("*.tif"))
results  = []

for fichier in tqdm(fichiers, desc="Traitement des fichiers", unit="fichier"):

    # 1️⃣ Meilleur plan de focus
    img_best = Include.Best_plan_determination(fichier, False)

    # 2️⃣ Stats image originale
    area, pmean, pmin, pmax = Include.Extract_fluo_informations(img_best)

    # 3️⃣ Stats image sans soma + masque
    img_true, mask, area_true, pmean_true, pmin_true, pmax_true = \
        Include.Extract_fluo_informations_without_nucleus(
            img_best, distance_transform, margin, False
        )
    Include.Save_images_and_overlay(output_folder, fichier, img_best, mask)

    # 4️⃣ Détection somas (CNN sur GPU)
    all_probs, soma_patches = Include.CNN_Patches_Construction(
        img_best, pretraitement_sigma, clip_limit,
        patch_size, cnn_threshold, model_cnn,
        batch_size=cnn_batch_size,
    )

    # 5️⃣ Clustering + fusion ROI
    merged_rois            = Include.Clusterization_detected_patches(img_best, soma_patches, patch_size)
    final_patches, filtered_rois = Include.Finale_Fusion_patches(img_best, merged_rois, overlap_thresh)

    # 6️⃣ Comptage lysosomes (Cellpose sur GPU)
    total_lysosomes = Include.Cellpose_Analyse_Count(
        cellpose_model, final_patches,
        diameter, cellprob_threshold, flow_threshold,
        min_area, max_area, min_circularity, max_axis_ratio,
        top_hat_radius, False,
        cellpose_batch_size=cellpose_batch_size,
    )

    # 7️⃣ Sauvegarde image annotée
    Include.Save_final_patches(img_best, filtered_rois, soma_patches,
                                   output_folder, fichier.name, half)

    # 8️⃣ Collecte résultats
    results.append({
        "file":       fichier.name,
        "area":       area,       "pmean":       pmean,
        "pmin":       pmin,       "pmax":        pmax,
        "area_true":  area_true,  "pmean_true":  pmean_true,
        "pmin_true":  pmin_true,  "pmax_true":   pmax_true,
        "n_lyso":     total_lysosomes,
    })

    # 9️⃣ Libération mémoire GPU entre chaque image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════
# 5. EXPORT CSV
# ═══════════════════════════════════════════════════════════════════════════

Include.Creation_CSV(results, output_folder)
