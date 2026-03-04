# Masterprojekt 1 — ToF/Structured Light Sensor Fusion for MPI Reduction

## Project Overview

This master's project investigates the combination of **indirect Time-of-Flight (iToF)** and **Structured Light (SL)** depth sensing to reduce the dominant systematic error of ToF cameras: **Multi-Path Interference (MPI)**. The long-term goal is to build a real hardware prototype and evaluate the fusion algorithm on actual sensor data. As a first step, both sensor modalities are simulated using a physically accurate renderer (PBRT), which allows full algorithmic development and validation before any hardware is built.

---

## The Core Problem: Multi-Path Interference in ToF Cameras

Indirect ToF cameras measure depth by emitting amplitude-modulated near-infrared light and computing the phase shift of the returning signal. The measured phase encodes the optical path length and thus the distance to the scene.

However, in real scenes, pixels do not receive light exclusively reflected from a single surface point. They also pick up **indirect (global) illumination** — light that has bounced off one or more additional surfaces before reaching the sensor. Since both the direct and indirect components are modulated at the same carrier frequency, the ToF sensor cannot distinguish between them. Their superposition produces a phase measurement that corresponds to a longer-than-true optical path, causing a systematic **depth overestimation** — this is MPI.

MPI is particularly severe in:
- **Room corners** — two planar reflectors create strong secondary reflections that contaminate nearby pixels
- **Concave surfaces** — curved surfaces focus indirect reflections towards the sensor
- **Translucent objects** — subsurface diffusion in materials like skin, wax, or plastic scatters light inside the object, further increasing the apparent path length

MPI is a fundamental limitation of the iToF principle and cannot be corrected by any single-modality approach without additional information.

---

## Why Structured Light Helps

Structured light (SL) recovers depth from **geometric triangulation**: a known pattern is projected from a known position, and its displacement in the camera image is used to compute depth via the epipolar geometry. This measurement is **fundamentally different from phase-based ToF**: it measures where light hits a surface, not how long it took. Therefore, SL is **immune to MPI by design** — the triangulation result is unaffected by indirect reflections.

The combination of both modalities leverages their complementarity:

| Property | iToF | Structured Light |
|---|---|---|
| Dense depth map | Yes (full-resolution) | Yes (sparse, dot-based) |
| MPI susceptibility | High | None |
| Accuracy at long range | Good | Degrades (disparity shrinks) |
| Accuracy at short range | Good | Excellent |
| Robustness | High | Medium (needs good contrast) |
| Subsurface diffusion errors | Yes (skin, wax, etc.) | None |

The hypothesis is that when iToF and SL agree on a depth measurement, the iToF value is trustworthy (not significantly corrupted by MPI). When they disagree, MPI is likely present, and the SL value (or a statistically optimal combination) should be preferred.

---

## Theoretical Foundation

The algorithm is primarily based on two papers:

### Paper 1 — "An iToF/Triangulation Depth Sensor for Mixed Reality Applications" (Godbaz et al., 2025, Microsoft)
*→ referred to as "Microsoft-Paper" in this project*

This paper describes a compact mixed-reality depth sensor that combines an iToF array (ADI ADSD3030) with a near-infrared dot projector in a single module (33.5 × 15 × 7.2 mm). It is the **primary foundation** for the calibration strategy and the core fusion metric used in this project.

Key contributions relevant to this project:

**Dot Calibration:**
A dedicated offline per-unit calibration procedure determines:
- **V**: A unit vector per dot (in transmitter space), describing the direction of each projected dot ray
- **AB**: The physical baseline between the dot projector (transmitter B) and the ToF sensor (receiver A)

The calibration requires a flat, diffuse target imaged at multiple distances. For each frame, dot positions are found in the intensity image using Laplacian-of-Gaussian (LoG) blob detection followed by Gaussian Process Regression (GPR) for subpixel localization. The iToF depth at each dot is back-projected to 3D using the camera intrinsics, yielding a set of 3D points per dot across all distances. Fitting a line through these points gives the dot's unit vector V; the intersection of all such lines gives the transmitter position B (and thus AB).

**Consistency Error:**
The key metric for fusing both modalities at runtime is the **consistency error** ε:

```
ε = | 1/Z_ToF − 1/Z_triangulation |
```

This is the absolute disagreement between the iToF depth and the triangulated SL depth, expressed in 1/Z space (which is proportional to image-space disparity along the epipolar line). A small ε indicates both sensors agree — the measurement is reliable and MPI-free. A large ε indicates a discrepancy, typically caused by MPI in the iToF measurement.

**Active Brightness Trail** *(planned for implementation)*:
The "dot trail" is the precomputed 1D path a projected dot traces across the image as target distance varies (from ~30 cm to infinity). At runtime, the active brightness image is resampled along this 1D trail instead of using 2D blob detection. This dramatically improves robustness at low SNR and in textured scenes, and also allows detection of MPI-inducing scene structures by identifying multiple brightness peaks along the trail.

### Paper 2 — "Combination of Spatially-Modulated ToF and Structured Light for MPI-Free Depth Estimation" (Agresti & Zanuttigh, ECCV 2018)

This paper proposes a different hardware approach: a ToF camera whose projector emits spatially modulated sinusoidal patterns instead of uniform flood illumination. Using Fourier analysis of the 9-sample ToF correlation sequence, two independent pieces of information are extracted from the **same single acquisition**:
1. An MPI-corrected ToF depth (direct/global light separation via high-frequency spatial modulation)
2. A structured light depth (from the recovered pattern phase offset θ)

These are then fused via **Maximum Likelihood (ML) estimation**, using analytically derived, per-pixel noise variance maps as weights:

```
d_fus(i,j) = argmax_Z  P(I_ToF(i,j)|Z) · P(I_SL(i,j)|Z)
```

Each likelihood term is modeled as a spatially-weighted mixture of Gaussians over a 7×7 pixel neighborhood, weighted by the inverse of the per-pixel depth variance. This produces a fused depth map that significantly outperforms either modality alone (mean absolute error of ~14 mm vs. ~50–65 mm for individual methods on synthetic Blender scenes).

This Agresti approach is implemented in the project as the second fusion strategy, operating on top of the separately acquired iToF and SL data.

---

## What Has Been Implemented

All current development is simulation-based. Both the SL images (EXR renders) and the ToF point clouds (PCD files) are generated using the PBRT physically-based renderer, producing realistic synthetic sensor data with controlled ground truth.

### 1. Simulation Setup (PBRT)

- SL images: rendered as `.exr` files at 10 target distances (0.4 m to 4.0 m in 0.4 m steps) — simulating the SL camera view of a flat dot-illuminated target
- ToF data: rendered as `.pcd` point clouds — simulating the ToF sensor's 3D measurements at the same distances
- Test scenes: separate SL and ToF captures of an arbitrary test target (not used for calibration)
- All data resides in `Simulation_Pictures/PBRT/SL_ToF_1/`

### 2. Calibration Pipeline (Based on Microsoft-Paper Paper)

Implemented in `Dot_Calibration_new.py` (class `DotCalibration`) and executed in the Jupyter notebook (`DotCalibrationNB_neueSimu.ipynb`).

**Step 1 — Image Loading:**
- Supports EXR images (OpenCV with `OPENCV_IO_ENABLE_OPENEXR`) and PCD point clouds (custom ASCII parser)
- Handles both organized and unorganized point clouds; reprojects to 2D depth maps via camera intrinsics K

**Step 2 — Dot Detection (LoG):**
- `LoG_blob_detection()`: Laplacian-of-Gaussian blob detector (scikit-image) applied to each SL image
- Returns blob centers and radii at the scale that maximizes LoG response

**Step 3 — Subpixel Localization:**
- `detectSubPixelLocation()` supports four subpixel refinement modes:
  - `"GPR"`: Gaussian Process Regression (most accurate, slowest)
  - `"center"`: Radial symmetry center (fast, robust)
  - `"geometricCenter"`: Flood-fill centroid around detected peak (robust to noise)
  - `"radial"`: Gradient-based radial symmetry
- Assigns stable integer IDs (0–99 for a 10×10 dot grid) sorted top-to-bottom, left-to-right for consistent cross-distance tracking

**Step 4 — Dot Identification & 3D Back-Projection:**
- `dot_identification()`: For each distance and each dot, the ToF depth at the detected pixel is used to compute the 3D point U_ij via back-projection through the camera intrinsics
- This yields a (100 dots × 10 distances × 3 coords) array U of calibration 3D points

**Step 5 — Unit Vector and Baseline Estimation:**
- For each dot i, a 3D line is fitted through all U_ij points using least-squares (parametrized by angle θ, φ to enforce unit-vector constraint)
- The baseline AB is recovered as the best-fitting intersection point of all dot ray lines in transmitter space
- Output: scalar AB (meters), array V of shape (100, 3) — one unit vector per dot

### 3. Fusion Approach 1 — Consistency Error (Microsoft-Paper Paper)

At test time, the detected dot positions in the test SL image are matched against the calibrated unit vectors V. For each detected dot at pixel (u, v):
- The iToF depth Z_ToF is sampled at that pixel location (via organized depth map or KD-tree nearest-neighbor fallback)
- The SL depth Z_triangulation is computed by solving the triangulation equation: given the camera ray r at (u, v), the dot unit vector V_i, and baseline B, find the depth that satisfies `s·r − t·V_i = B` in least-squares sense
- The consistency error ε = |1/Z_ToF − 1/Z_triangulation| is computed; a large ε flags likely MPI corruption

### 4. Fusion Approach 2 — Maximum Likelihood Fusion (Agresti Paper)

Implemented as a second approach. The ML fusion combines the ToF and SL depth estimates with weights derived from the per-pixel noise variance of each modality. The spatially-weighted mixture-of-Gaussians likelihood is maximized over a restricted candidate set (within 3σ of both estimates), producing a fused depth value per dot that is statistically optimal under the assumed noise model.

---

## Repository Structure

```
Masterprojekt_1/
├── code/
│   ├── Dot_Calibration_new.py          # Core calibration library (DotCalibration class)
│   ├── DotCalibrationNB_neueSimu.ipynb # Main pipeline notebook
│   └── csvEdit.py                      # Utility for CSV data editing
├── Simulation_Pictures/
│   └── PBRT/
│       └── SL_ToF_1/
│           ├── SL/                     # Simulated SL renders (.exr), calibration distances
│           ├── ToF/                    # Simulated ToF point clouds (.pcd), calibration distances
│           └── General_Test/           # Test scene (Test_SL.exr, Test_ToF.pcd)
├── Papers/
│   ├── Microsoft-Paper_*.pdf                      # Godbaz et al. 2025 (Microsoft) — primary reference
│   └── Agresti_*.pdf                   # Agresti & Zanuttigh, ECCV 2018 — secondary reference
└── Description.md                      # This document
```

---

## Key Variables and Notation

| Symbol | Shape | Description |
|---|---|---|
| **K** | 3×3 | Camera intrinsics matrix (fx=fy=483 px, cx=320, cy=240) |
| **AB** | scalar | Baseline distance between projector (Tx) and camera (Rx), in meters |
| **B** | (3,) | Transmitter position in camera coords: [AB, 0, 0] |
| **V** | (100, 3) | Unit vectors of all calibrated dots in transmitter space |
| **U** | (100, n_dist, 3) | 3D back-projected calibration points per dot per distance |
| **U_tx** | (100, n_dist, 3) | U expressed in transmitter reference frame (U − B) |
| **subpixel_list** | list of dicts | Per-distance list of {id, x, y} subpixel coordinates |
| **ε** | scalar | Consistency error: \|1/Z_ToF − 1/Z_SL\| |
| **σ²_ToF** | per-pixel | iToF depth noise variance (analytical from shot-noise model) |
| **σ²_SL** | per-pixel | SL depth noise variance (proportional to d², inversely to baseline) |
