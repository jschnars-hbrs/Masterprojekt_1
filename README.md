<h1>Master's Project: 1 - Combining Structured Light and iToF</h1>

This repository contains code for my master's project. The main goal is to combine Structured Light measurements with (usually more reliable / accurate) 
iToF measurements. 

<h2>References</h2>

The codebase is mainly based on two papers:
1. ["An iToF/triangulation depth sensor for mixed reality applications"](https://doi.org/10.1117/12.3040965) - Godbaz et al., SPIE 2025
2. ["Combination of Spatially-Modulated ToF and Structured Light for MPI-Free Depth Estimation"](https://doi.org/10.1007/978-3-030-11009-3_21) - Agresti & Zanuttigh, ECCV 2018

<h2>Programming Language</h2>

The code is fully written in Python. If the runtime gets too high, it may be rewritten in a more performant language, but for the foreseeable future, it
will stay in python. 

<h2>Simulation</h2>

The Structured-Light-System and the iToF-System have been simulated via [PBRT](https://github.com/mmp/pbrt-v4).
Since PBRT does not offer iToF-Simulation as a native feature, it has been implemented by our working group. I am using this implementation for 
simulating.
In the next 1-2 months it is planned, to build an experimental setup, with a DOE + VCSEL as a Structured Light projector, combined with an iToF-Camera.

<h2> Approaches: Combining SL + iToF</h2>

Currently, there are 3 different approaches implemented, how SL + iToF is being combined:

<h3>Approach 1</h3>

This approach is based on the first referenced paper.
The steps are as follows:
1. Locate the dots in the dot pattern, using LoG-Blob-Detection (Laplacian of Gaussian)
2. Find the subpixel peak of the rough estimate blob (With -> Centroid, Center or GPR (Gaussian Process Regression))
3. Compute Triangulation for the subpixel locations
4. Find the lowest  $\epsilon = | \frac{1}{Z_{ToF}} - \frac{1}{Z_{SL}}|$ , with the help of the iToF distances
5. Return the iToF distance at the lowest $\epsilon$

<h3>Approach 2</h3>

This approach is based on the second referenced paper.

The steps are as follows:
1. Locate the dots in the dot pattern, using LoG-Blob-Detection (Laplacian of Gaussian)
2. Find the subpixel peak of the rough estimate blob (With -> Centroid, Center or GPR (Gaussian Process Regression))
3. Compute Triangulation for the subpixel locations
4. For Triangulation and iToF Measurments, compute Maximum-Likelihood-Function -> The probability of the measured distance is a Gaussian ($\sigma$ for ToF is lower than for SL -> ToF is generally more accurate)
5. Find the argmax of the combined Maximum-Likelihood-Functions and return the depth, found by argmax


<h3>Approach 3</h3>

This approach is based on the first referenced paper and aims to mitigate MPI effects.

The steps are as follows:
1. Place epipolar lines through the dot trails (in a 10x10 pattern, there a 10 rows, so 10 epipolar lines / trails)
2. Find Peaks in the Brightness of the picture
3. Evaluate every Peak and find lowest $\epsilon = | \frac{1}{Z_{ToF}} - \frac{1}{Z_{SL}}|$, with the help of iToF
4. Find subsample location using a 1D quadratic fit
5. Triangulate with found location and return the triangulated depth

<h2>Code Structure and Usage</h2>

The code is divided into two files.
1. dot_calibration.py - A class where fundamental functions, which are being repeatedly used, located.
2. iToF_SL_fusion_pipeline.ipynb - Jupyter Notebook where the calibration is being executed and implementation of the 3 different Approaches.

What's needed to run the code:
1. requirements.txt
2. SL (png or exr) and iToF-Simulation (pcd or csv) -> Calibration + Test files
