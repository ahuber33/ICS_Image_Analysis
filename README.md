# ICS Image Analysis

**Automated quantification of lysosomes in neuronal dendrites from fluorescence microscopy images.**

Developed at **LP2I Bordeaux** — Arnaud HUBER, Research Engineer  
Contact : huber@lp2ib.in2p3.fr

---

## Quick Start — Get the code

**GitHub:**
```bash
git clone https://github.com/ahuber33/ICS_Image_Analysis.git
```

Then move into the project folder:
```bash
cd ICS_Image_Analysis
```


Once cloned, open `Analyse.py`, set the path to your image folder, and follow the [Installation](#4-installation) section below.

---

## Table of Contents

1. [Scientific Context](#1-scientific-context)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Project Structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Parameters Reference](#6-parameters-reference)
7. [Output Files](#7-output-files)
8. [FAQ / Troubleshooting](#8-faq--troubleshooting)

---

## 1. Scientific Context

This tool was developed to study the impact of heavy metals — particularly **Cadmium (Cd)** — on neuronal dendrites. Neurons are imaged by fluorescence microscopy using two channels:

| Channel | Fluorophore | Target |
|---------|-------------|--------|
| `w1TRITC` | TubulinA594 (red) | Neuronal cytoskeleton / dendrites |
| `w2GFP` | pTau488 or Tau488 (green) | Tau protein (dendritic marker) |

Each experiment compares control neurons against neurons exposed to Cadmium for varying durations (e.g. 48h). The pipeline automatically:

- Selects the sharpest focal plane from each multi-plane TIFF stack
- Detects neuronal somas (cell bodies) using a CNN classifier
- Counts lysosomes within each soma using Cellpose segmentation
- Exports quantitative statistics per image and per channel to CSV

---

## 2. Pipeline Overview

```
TIFF stack (Z planes)
        │
        ▼
 Best focus plane selection (variance-based)
        │
        ▼
 Fluorescence statistics (full image + soma-excluded region)
        │
        ▼
 Soma detection — CNN (SomaCNN) + blob detection (blob_dog)
        │
        ▼
 Patch clustering + ROI fusion
        │
        ▼
 Lysosome segmentation per soma — Cellpose (cyto3)
        │
        ▼
 CSV export + annotated images
```

---

## 3. Project Structure

The repository contains only the code. Image data and outputs are stored wherever you choose on your machine — the path is configured directly in `Analyse.py`.

```
ics-image-analysis/
│
├── Include.py              # All analysis functions (library)
├── Analyse.py              # Main script — batch processing loop
└── soma_cnn_test.pth       # Pre-trained CNN weights (soma detection)
```

When the analysis runs, an `output/` subfolder is **created automatically** inside your image folder:

```
<your_image_folder>/
│
└── output/
    ├── Results_with_true.csv
    ├── *_best.png
    ├── *_mask_overlay.png
    └── *_annotated.png
```

### Input file naming convention

Each `.tif` file contains a **multi-plane Z-stack** (typically 7 planes). The channel is identified by the suffix in the filename:

| Suffix | Channel |
|--------|---------|
| `_w1TRITC` | TubulinA594 — cytoskeleton |
| `_w2GFP` | pTau488 or Tau488 — Tau protein |

The pipeline automatically selects the sharpest Z-plane from each stack.

---

## 4. Installation

### 4.1 Prerequisites

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.12.8 | Via Anaconda |
| NVIDIA GPU | any | Optional but strongly recommended |
| CUDA driver | ≥ 11.8 | Only if using GPU — see step below |
| Spyder IDE | any | Recommended for interactive use |
| Internet connection | — | Required on first run to download the Cellpose cyto3 model (~200 MB) |

> ℹ️ **The pipeline runs on CPU if no GPU is available**, but will be significantly slower on large batches.  
> GPU support requires an NVIDIA card. AMD and Intel GPUs are not supported by PyTorch/Cellpose.

---

### 4.2 Check your CUDA version (GPU users only)

Open a terminal and run:

```
nvidia-smi
```

Look for **`CUDA Version`** in the top-right corner of the output. Note this number — you will need it in the next step.

If `nvidia-smi` is not found, your NVIDIA driver is not installed. Download it from [nvidia.com/drivers](https://www.nvidia.com/drivers) before continuing.

---

### 4.3 Option A — Spyder / Anaconda (recommended)

Open the **Anaconda Prompt** and create a dedicated environment:

```bash
conda create -n ics_analysis python=3.12.8
conda activate ics_analysis
```

Install Spyder in the environment:

```bash
conda install spyder
```

Install PyTorch — **choose the command that matches your CUDA version**:

```bash
# CUDA 12.4 or higher (most recent drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# No GPU / CPU only
pip install torch torchvision
```

> ⚠️ The CUDA version in the pip command must be **less than or equal to** the version shown by `nvidia-smi`.  
> For example: if `nvidia-smi` shows `CUDA Version: 12.6`, install `cu124` (not cu126, which doesn't exist yet).

Install all other dependencies with exact versions:

```bash
pip install cellpose==2.1.0
pip install scikit-image==0.25.2
pip install tifffile==2023.4.12
pip install scipy==1.17.1
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install tqdm
pip install matplotlib
```

Launch Spyder from within the environment:

```bash
spyder
```

> ⚠️ Always launch Spyder **from the Anaconda Prompt after activating the environment**, not from the Windows Start menu, to ensure the correct Python interpreter is used.

---

### 4.4 Option B — General environment (pip / virtualenv)

```bash
python -m venv ics_env
# Windows:
ics_env\Scripts\activate
# Linux/macOS:
source ics_env/bin/activate

# Replace cu124 with your CUDA version (cu121, cu118) or remove --index-url for CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install cellpose==2.1.0 scikit-image==0.25.2 tifffile==2023.4.12 \
            scipy==1.17.1 numpy==1.26.4 pandas==2.2.2 tqdm matplotlib
```

---

### 4.5 Verify GPU activation

Run this in Python to confirm everything is correctly installed:

```python
import torch
from cellpose import models

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name        :", torch.cuda.get_device_name(0))

model = models.Cellpose(gpu=torch.cuda.is_available(), model_type='cyto3')
print("Cellpose GPU    :", model.gpu)
```

Expected output **(with GPU)**:
```
PyTorch version : 2.6.0+cu124
CUDA available  : True
GPU name        : <your GPU name>
Cellpose GPU    : True
```

Expected output **(CPU only)**:
```
PyTorch version : 2.6.0+cpu
CUDA available  : False
Cellpose GPU    : False
```

> If `CUDA available` shows `False` despite having an NVIDIA GPU, see the [FAQ](#8-faq--troubleshooting).

---

## 5. Usage

### 5.1 Configure the input path

Open `Analyse.py` and set the path to your image folder:

```python
dossier = Path(r'D:\ICS\your_experiment_folder\Condition1')
```

### 5.2 Adjust batch sizes for your GPU (if needed)

```python
cnn_batch_size      = 256   # reduce to 128 if CUDA out-of-memory error
cellpose_batch_size = 8     # reduce to 4 if CUDA out-of-memory error
```

### 5.3 Run the analysis

In Spyder, open `Analyse.py` and press **F5** (Run).

A progress bar will display the processing status:

```
Traitement des fichiers: 100%|██████████| 24/24 [02:13<00:00]
```

Results are saved automatically in the `output/` subfolder of your image directory.

---

## 6. Parameters Reference

### Pre-processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretraitement_sigma` | `1.5` | Gaussian blur sigma before blob detection |
| `clip_limit` | `0.02` | CLAHE contrast enhancement clip limit |

### Soma detection (CNN)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cnn_threshold` | `0.23` | Minimum CNN probability to classify a patch as soma |
| `patch_size` | `64` | Size (px) of the square patch extracted around each blob |
| `cnn_batch_size` | `256` | Patches per CNN forward pass (increase for faster processing) |

### Soma exclusion (fluorescence stats)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_transform` | `40` | Distance (px) to erode soma mask before exclusion |
| `margin` | `40` | Extra margin (px) added around each soma bounding box |

### ROI fusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overlap_thresh` | `0.9` | Fraction of overlap above which a smaller ROI is discarded |

### Cellpose segmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diameter` | `8` | Expected lysosome diameter in pixels |
| `cellprob_threshold` | `0.1` | Cellpose cell probability threshold |
| `flow_threshold` | `0.5` | Cellpose flow error threshold |
| `min_area` | `5` | Minimum area (px²) to keep a detected object |
| `max_area` | `500` | Maximum area (px²) to keep a detected object |
| `min_circularity` | `0.5` | Minimum circularity (0–1) — filters elongated dendrite fragments |
| `max_axis_ratio` | `2` | Maximum ratio of major/minor axis — filters dendrites |
| `top_hat_radius` | `10` | Disk radius for white top-hat pre-filter |
| `cellpose_batch_size` | `8` | Patches per Cellpose forward pass |

---

## 7. Output Files

### `Results_with_true.csv`

Main results table, one row per image, with columns for both channels (`w1TRITC` and `w2GFP`):

| Column | Description |
|--------|-------------|
| `file` | Original filename |
| `area` | Total image area (pixels) |
| `pmean` | Mean pixel intensity (full image) |
| `pmin` / `pmax` | Min / max pixel intensity |
| `area_true` | Area after soma exclusion |
| `pmean_true` | Mean intensity after soma exclusion |
| `pmin_true` / `pmax_true` | Min / max after soma exclusion |
| `n_lyso` | Number of lysosomes detected |

> The CSV uses `;` as column separator and `,` as decimal separator (compatible with French Excel).

### `*_best.png`

The sharpest Z-plane selected from the input TIFF stack, saved as a grayscale PNG.

### `*_mask_overlay.png`

The best-focus image with the soma exclusion mask overlaid in red. Useful for visually validating that somas are correctly identified.

### `*_annotated.png`

The best-focus image with detected soma patches drawn as coloured rectangles. Rectangle colour encodes the average CNN confidence score (blue = low, yellow = high, viridis scale).

---

## 8. FAQ / Troubleshooting

**`torch.cuda.is_available()` returns `False`**  
→ Your PyTorch installation does not match your CUDA driver. Run `nvidia-smi` and check the `CUDA Version`. Then reinstall PyTorch:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

**`CUDA out of memory` error**  
→ Reduce batch sizes in `Analyse.py`:
```python
cnn_batch_size      = 128
cellpose_batch_size = 4
```

---

**No somas detected on an image**  
→ Try lowering `cnn_threshold` (e.g. from `0.23` to `0.15`). If the image quality is poor, check the `*_best.png` output to verify focus plane selection.

---

**Too many or too few lysosomes counted**  
→ Adjust Cellpose parameters: increase `min_area` to remove noise, decrease `diameter` if lysosomes appear smaller than expected, or tune `min_circularity` to filter dendrite fragments more aggressively.

---

**`FileNotFoundError: soma_cnn_test.pth`**  
→ The CNN weights file must be placed in the **same folder as `Analyse.py`**, inside `Analyse/`. Do not rename it.

---

**Cellpose cyto3 model download**  
→ The Cellpose cyto3 model (~200 MB) is **downloaded automatically** on the first run. An internet connection is required only for this initial download. It is then cached locally in `C:\Users\<your_name>\.cellpose\models\` and reused for all subsequent runs.

---

**`HTTPError: HTTP Error 500: INTERNAL SERVER ERROR` when loading Cellpose**  
→ This is a known Cellpose 2.x issue: the server sometimes fails to serve the model files. Two options to fix it:

**Option 1 — Direct download from cellpose.org** (fastest)

Download the following two files directly in your browser:
- `https://www.cellpose.org/models/cyto3`
- `https://www.cellpose.org/models/size_cyto3.npy`

**Option 2 — Download from BioImage.IO Model Zoo**

1. Go to **[https://bioimage.io/#/artifacts/famous-fish](https://bioimage.io/#/artifacts/famous-fish)**
2. Click the download icon on the model card
3. Select **"Download by Weight Format"** → **"Pytorch State Dict"**

**After downloading (both options):**

Place the files in the Cellpose model cache folder:
- **Windows** : `C:\Users\<your_name>\.cellpose\models\`
- **Linux / macOS** : `/home/<your_name>/.cellpose/models/`

The files must be named exactly `cyto3` and `size_cyto3.npy` (no extension for the first one).

> On Windows, the `.cellpose` folder may be hidden. In File Explorer, enable **View > Hidden items** to see it.

Re-run the script — Cellpose will find the files locally and skip the download.

---

**Spyder uses the wrong Python environment**  
→ Always launch Spyder from the Anaconda Prompt after activating the environment:
```bash
conda activate ics_analysis
spyder
```
In Spyder, verify via `Tools > Preferences > Python interpreter` that it points to the `ics_analysis` environment.

---

**The CSV opens incorrectly in Excel**  
→ The file uses `;` separators and `,` decimal — this is the standard for French Excel. If using English Excel, go to `Data > From Text/CSV` and manually set the delimiter to `;`.

---

*ICS Image Analysis — LP2I Bordeaux — Arnaud HUBER*
