import os
import re
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from skimage import feature
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List, Dict


class DotCalibration:
    """
    Calibration and depth fusion for a combined iToF + dot projector sensor.

    Based on:
      - Godbaz et al. 2025 (Microsoft-Paper): dot calibration, consistency error, active brightness trail
      - Agresti & Zanuttigh, ECCV 2018: maximum likelihood depth fusion
    """

    # ─────────────────────────────────────────────────────────────────────────
    # File utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def parse_distance_from_name(filename: str) -> Optional[float]:
        """
        Parse target distance in metres from a filename.

        Handles formats like:  SL_0.4m.exr, SL_ToF_1.2m.pcd, 3_6m_frame.png
        """
        m = re.search(r"([0-9]+(?:[.,_][0-9]+)?)m", os.path.basename(filename), re.IGNORECASE)
        if not m:
            return None
        s = m.group(1).replace("_", ".").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None

    def load_images(self, folder_path: str, pattern: str = "_image_rendered.png") -> List[str]:
        """Return sorted list of image paths matching *pattern* inside *folder_path*."""
        image_paths = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and file.endswith(pattern):
                image_paths.append(full_path)

        image_paths.sort(key=lambda p: self.parse_distance_from_name(p) or 999.0)
        return image_paths

    def read_image(self, image_path: str) -> np.ndarray:
        """Read an image (EXR or PNG) and return it as a grayscale array."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # ─────────────────────────────────────────────────────────────────────────
    # ToF data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_pcd_header(self, pcd_path: str) -> dict:
        """Parse PCD ASCII header, return metadata dict."""
        meta = {}
        header_len = 0
        with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                header_len += 1
                s = line.strip()
                if not s:
                    continue
                if s.startswith("FIELDS"):
                    meta["fields"] = s.split()[1:]
                elif s.startswith("WIDTH"):
                    meta["width"] = int(s.split()[1])
                elif s.startswith("HEIGHT"):
                    meta["height"] = int(s.split()[1])
                elif s.startswith("POINTS"):
                    meta["points"] = int(s.split()[1])
                elif s.startswith("DATA"):
                    meta["data"] = s.split()[1].lower()
                    break
        meta["header_len"] = header_len
        if "data" not in meta:
            raise ValueError(f"PCD header incomplete – no DATA field in {pcd_path}")
        return meta

    def load_tof_pcd(self, pcd_path: str, unit_scale: float = 0.001,
                     depth_mode: str = "radial") -> dict:
        """
        Load ToF data from an organised PCD (ASCII) file.

        Parameters
        ----------
        pcd_path    : path to .pcd file
        unit_scale  : multiply xyz by this factor (0.001 converts mm → m)
        depth_mode  : "radial" = sqrt(x²+y²+z²); "axial" or "z" = z only

        Returns
        -------
        tof_data dict with keys:
          points_3d, distance, intensity, width, height
          + points_map (H,W,3), depth_map (H,W), intensity_map (H,W) for organised clouds
        """
        meta = self._parse_pcd_header(pcd_path)
        if meta["data"] != "ascii":
            raise NotImplementedError(f"Only DATA ascii supported (got: {meta['data']})")

        fields = meta.get("fields")
        if not fields:
            raise ValueError(f"No FIELDS in PCD header: {pcd_path}")

        df = pd.read_csv(pcd_path, sep=r"\s+", header=None, names=fields,
                         skiprows=meta["header_len"], engine="c").dropna(how="all")

        for c in ("x", "y", "z"):
            if c not in df.columns:
                raise ValueError(f"PCD missing column '{c}' (FIELDS={fields})")

        points = df[["x", "y", "z"]].to_numpy(dtype=np.float64) * unit_scale

        if "grayValue" in df.columns:
            intensity = df["grayValue"].to_numpy(dtype=np.float64)
        elif "intensity" in df.columns:
            intensity = df["intensity"].to_numpy(dtype=np.float64)
        else:
            intensity = np.full(len(df), np.nan, dtype=np.float64)

        if depth_mode == "radial":
            distance = np.linalg.norm(points, axis=1)
        elif depth_mode in ("z", "axial"):
            distance = points[:, 2]
        else:
            raise ValueError("depth_mode must be 'radial' or 'axial'")

        tof_data = {
            "points_3d": points,
            "distance": distance,
            "intensity": intensity,
            "width": meta.get("width"),
            "height": meta.get("height"),
        }

        W, H = tof_data["width"], tof_data["height"]
        if W is not None and H is not None and W * H == len(points):
            tof_data["points_map"]   = points.reshape((H, W, 3))
            tof_data["depth_map"]    = distance.reshape((H, W))
            tof_data["intensity_map"] = intensity.reshape((H, W))

        return tof_data

    def load_tof_csv(self, csv_path: str, use_noise: bool = True) -> dict:
        """
        Load simulated ToF data from a semicolon-delimited CSV (Blender export).

        Expected columns: X, Y, Z, distance [, X_noise, Y_noise, Z_noise, distance_noise, intensity]
        """
        df = pd.read_csv(csv_path, sep=";")
        tof_data = {}

        if use_noise and "X_noise" in df.columns:
            tof_data["points_3d"]   = df[["X_noise", "Y_noise", "Z_noise"]].values
            tof_data["distance"]    = df["distance_noise"].values
            tof_data["points_3d_gt"] = df[["X", "Y", "Z"]].values
            tof_data["distance_gt"]  = df["distance"].values
        else:
            tof_data["points_3d"] = df[["X", "Y", "Z"]].values
            tof_data["distance"]  = df["distance"].values

        tof_data["intensity"] = df["intensity"].values if "intensity" in df.columns else np.full(len(df), np.nan)
        return tof_data

    # ─────────────────────────────────────────────────────────────────────────
    # Blob detection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def add_gaussian_to_detected_blob(image: np.ndarray, blobs: np.ndarray) -> np.ndarray:
        """Add a synthetic 2-D Gaussian at each blob location to create Gaussian intensity profiles."""
        GAUSS_SIGMA = 2   # fixed narrow sigma in pixels
        GAUSS_R     = 4     # half-window size in pixels
        out = image.astype(np.float64).copy()
        H, W = out.shape[:2]
        for blob in blobs:
            y, x = blob[0], blob[1]
            y0, y1 = max(0, int(y) - GAUSS_R), min(H, int(y) + GAUSS_R + 1)
            x0, x1 = max(0, int(x) - GAUSS_R), min(W, int(x) + GAUSS_R + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gauss = np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / GAUSS_SIGMA ** 2)
            vy = int(np.clip(round(y), 0, H - 1))
            vx = int(np.clip(round(x), 0, W - 1))
            peak = float(image[vy, vx])
            out[y0:y1, x0:x1] += peak * gauss
        return out

    def detect_blobs(self, image_path: str, max_sigma: int = 30, num_sigma: int = 10,
                     threshold: float = 0.1, visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Laplacian-of-Gaussian blob detector (scikit-image).

        Returns
        -------
        blobs : (N, 3) array of (y, x, radius)
        image : grayscale image array with synthetic Gaussians added at each blob
        """
        image = self.read_image(image_path)
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image for LoG blob detection: {image_path}")

        blobs = feature.blob_log(image, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        blobs[:, 2] = blobs[:, 2] * (2 ** 0.5)  # convert sigma to radius
        image_out = self.add_gaussian_to_detected_blob(image, blobs)

        if visualize:
            fig, ax = plt.subplots()
            ax.imshow(image_out, cmap="gray")
            for y, x, r in blobs:
                ax.add_patch(plt.Circle((x, y), r, color="red", linewidth=1, fill=False))
            plt.show()

        return blobs, image_out

    # ─────────────────────────────────────────────────────────────────────────
    # Subpixel localisation
    # ─────────────────────────────────────────────────────────────────────────

    def _gpr_peak(self, patch: np.ndarray) -> Tuple[float, float]:
        """GPR-based subpixel peak within *patch*. Returns (x, y) relative to patch."""
        h, w = patch.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        X = np.column_stack([xx.ravel(), yy.ravel()])
        y = patch.ravel().astype(float)

        kernel = 1.0 * RBF(length_scale=20.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
        gp.fit(X, y)
        y_mean = gp.predict(X)
        idx = int(np.argmax(y_mean))
        return float(X[idx, 0]), float(X[idx, 1])

    def _radial_symmetry_center(self, patch: np.ndarray, eps: float = 1e-9) -> Tuple[float, float]:
        """Radial-symmetry subpixel centre. Returns (x, y) relative to patch."""
        I = patch.astype(np.float64)
        I = I - np.median(I)
        I[I < 0] = 0

        h, w = I.shape
        gy, gx = np.gradient(I)
        mag = np.hypot(gx, gy) + eps
        ux, uy = gx / mag, gy / mag

        yy, xx = np.mgrid[0:h, 0:w]
        mask = mag > np.percentile(mag, 70)
        ux, uy = ux[mask], uy[mask]
        x, y   = xx[mask].astype(float), yy[mask].astype(float)
        wgt    = (mag[mask] ** 2).astype(float)

        A  = np.column_stack([-uy, ux])
        b  = -uy * x + ux * y
        W  = np.sqrt(wgt)
        sol, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        xc = float(np.clip(sol[0], 0, w - 1))
        yc = float(np.clip(sol[1], 0, h - 1))
        return xc, yc

    def _geometric_centroid(self, patch: np.ndarray, bg_percentile: int = 20,
                            frac: float = 0.35, peak_percentile: int = 99,
                            connectivity: int = 8) -> Tuple[float, float]:
        """Flood-fill centroid around the brightest spot. Returns (x, y) relative to patch."""
        h, w = patch.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        y_flat = patch.ravel().astype(float)

        idx_peak = int(np.argmax(y_flat))
        x_peak_i, y_peak_i = int(xx.ravel()[idx_peak]), int(yy.ravel()[idx_peak])

        bg  = float(np.percentile(y_flat, bg_percentile))
        pk  = float(np.percentile(y_flat, peak_percentile))
        thresh = bg + frac * (pk - bg)
        mask2d = patch.astype(float) >= thresh

        if not mask2d[y_peak_i, x_peak_i]:
            ys, xs = np.nonzero(mask2d)
            d2 = (xs - x_peak_i) ** 2 + (ys - y_peak_i) ** 2
            k  = int(np.argmin(d2))
            y_peak_i, x_peak_i = int(ys[k]), int(xs[k])

        neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] \
                if connectivity == 8 else [(-1,0),(1,0),(0,-1),(0,1)]

        visited = np.zeros_like(mask2d, dtype=bool)
        stack   = [(y_peak_i, x_peak_i)]
        visited[y_peak_i, x_peak_i] = True
        comp_xs, comp_ys = [], []

        while stack:
            cy, cx = stack.pop()
            if not mask2d[cy, cx]:
                continue
            comp_xs.append(cx)
            comp_ys.append(cy)
            for dy, dx in neigh:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if mask2d[ny, nx]:
                        stack.append((ny, nx))

        return float(np.mean(comp_xs)), float(np.mean(comp_ys))

    def detect_subpixel_locations(self, all_blobs: np.ndarray, image: np.ndarray,
                                   grid_cols: int = 10, flip_x: bool = False,
                                   flip_y: bool = False, mode: str = "GPR") \
            -> Tuple[List[np.ndarray], List[dict]]:
        """
        Refine blob positions to subpixel accuracy and assign stable dot IDs.

        Modes: "GPR" | "center" | "geometricCenter" | "radial"

        Returns
        -------
        patches    : list of image patches, one per dot
        subpixels  : list of dicts {id, x, y}, sorted top-left → bottom-right
        """
        H, W = image.shape[:2]
        blobs = np.asarray(all_blobs, dtype=float)
        entries = []

        for blob in blobs:
            y_c, x_c = int(blob[0]), int(blob[1])
            r = int(np.ceil(blob[2]))
            y0, y1 = max(0, y_c - r), min(H, y_c + r + 1)
            x0, x1 = max(0, x_c - r), min(W, x_c + r + 1)
            patch = image[y0:y1, x0:x1]

            if mode == "GPR":
                x_sub, y_sub = self._gpr_peak(patch)
            elif mode == "center":
                x_sub = float(patch.shape[1]) / 2.0
                y_sub = float(patch.shape[0]) / 2.0
            elif mode == "geometricCenter":
                x_sub, y_sub = self._geometric_centroid(patch)
            elif mode == "radial":
                x_sub, y_sub = self._radial_symmetry_center(patch)
            else:
                raise ValueError(f"Unknown subpixel mode: {mode}")

            entries.append({"x": float(x0 + x_sub), "y": float(y0 + y_sub), "patch": patch})

        n = len(entries)
        if n == 0:
            return [], []

        grid_cols = max(1, int(grid_cols))
        if n % grid_cols != 0:
            raise ValueError(f"Number of detected dots ({n}) does not match grid_cols={grid_cols}")

        grid_rows = n // grid_cols
        sorted_by_y = sorted(entries, key=lambda d: d["y"])
        if flip_y:
            sorted_by_y = list(reversed(sorted_by_y))

        ordered = []
        for r in range(grid_rows):
            row = sorted_by_y[r * grid_cols:(r + 1) * grid_cols]
            row = sorted(row, key=lambda d: d["x"], reverse=flip_x)
            ordered.extend(row)

        patches   = [e["patch"] for e in ordered]
        subpixels = [{"id": i, "x": e["x"], "y": e["y"]} for i, e in enumerate(ordered)]
        return patches, subpixels

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 3 – 3-D back-projection
    # ─────────────────────────────────────────────────────────────────────────

    def backproject_calibration_dots(self, subpixel_list: List[List[dict]],
                                      tof_paths: List[str], tof_mode: str = "pcd",
                                      K: Optional[np.ndarray] = None,
                                      pcd_unit_scale: float = 0.001,
                                      pcd_depth_mode: str = "radial",
                                      baseline_guess: float = 3.8e-2) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-project each detected dot position into 3-D using the ToF depth
        at that pixel, yielding calibration points U.

        Parameters
        ----------
        subpixel_list   : list (one entry per distance) of subpixel dicts {id, x, y}
        tof_paths       : list of ToF data file paths, same order as subpixel_list
        tof_mode        : "pcd" or "csv"
        K               : 3×3 camera intrinsics (default: fx=fy=483, cx=320, cy=240)
        pcd_unit_scale  : scale for PCD xyz (default 0.001 → mm to m)
        pcd_depth_mode  : "radial" or "axial"
        baseline_guess  : initial x-offset guess for transmitter [m], used to compute U_tx

        Returns
        -------
        U    : (n_dots, n_dist, 3) 3-D calibration points in camera coordinates [m]
        U_tx : (n_dots, n_dist, 3) U shifted into approximate transmitter space
        """
        if K is None:
            K = np.array([[483.0, 0.0, 320.0],
                          [0.0,  483.0, 240.0],
                          [0.0,   0.0,   1.0]])

        n_dots = len(subpixel_list[0])
        n_dist = len(subpixel_list)
        U = np.full((n_dots, n_dist, 3), np.nan, dtype=np.float64)

        for j, subpixels in enumerate(subpixel_list):
            if tof_mode.lower() == "pcd":
                tof_data = self.load_tof_pcd(tof_paths[j], unit_scale=pcd_unit_scale,
                                              depth_mode=pcd_depth_mode)
                if "depth_map" not in tof_data:
                    raise ValueError("PCD is not an organised cloud (no depth_map).")
                depth_map  = tof_data["depth_map"]
                points_map = tof_data["points_map"]
                H, W = depth_map.shape

            elif tof_mode.lower() == "csv":
                tof_data  = self.load_tof_csv(tof_paths[j], use_noise=True)
                depth_map = points_map = None
                distances = tof_data["distance"]

            else:
                raise ValueError("tof_mode must be 'pcd' or 'csv'")

            for dot in subpixels:
                i    = dot["id"]
                u_ij = dot["x"]
                v_ij = dot["y"]

                if tof_mode.lower() == "pcd":
                    u_int, v_int = int(round(u_ij)), int(round(v_ij))
                    if not (0 <= u_int < W and 0 <= v_int < H):
                        continue
                    d_check = depth_map[v_int, u_int]
                    if not (np.isfinite(d_check) and d_check > 1e-6):
                        continue
                    P = points_map[v_int, u_int, :]
                    if not np.all(np.isfinite(P)):
                        continue
                    U[i, j, :] = P

                else:  # csv
                    if i < 0 or i >= len(distances):
                        continue
                    d_ij = distances[i]
                    if not np.isfinite(d_ij) or d_ij <= 1e-6:
                        continue
                    K_inv = np.linalg.inv(K)
                    ray   = K_inv @ np.array([u_ij, v_ij, 1.0])
                    ray  /= np.linalg.norm(ray)
                    U[i, j, :] = d_ij * ray

        B_guess = np.array([baseline_guess, 0.0, 0.0])
        U_tx = U - B_guess[None, None, :]
        return U, U_tx

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 4 – Baseline estimation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def angles_to_unit_vector(theta: float, phi: float) -> np.ndarray:
        """Convert spherical angles (θ, φ) to a 3-D unit vector."""
        ct = np.cos(theta)
        return np.array([ct * np.cos(phi), ct * np.sin(phi), np.sin(theta)], dtype=float)

    @staticmethod
    def unit_vector_to_angles(v: np.ndarray) -> Tuple[float, float]:
        """Convert a 3-D vector to spherical angles (θ, φ)."""
        v = v / (np.linalg.norm(v) + 1e-12)
        theta = float(np.arcsin(np.clip(v[2], -1.0, 1.0)))
        phi   = float(np.arctan2(v[1], v[0]))
        return theta, phi

    def estimate_baseline(self, U_tx: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Estimate the transmitter position from U_tx (3-D points in approximate transmitter space).

        Fits a 3-D line through each dot's multi-distance samples, then finds the
        least-squares intersection of all lines.  The x-component of the intersection
        is the correction to the initial baseline guess.

        Parameters
        ----------
        U_tx : (n_dots, n_dist, 3) array of calibration points shifted by baseline_guess

        Returns
        -------
        intersection : (3,) intersection point in U_tx space
                       AB_correction = baseline_guess + intersection[0]
        n_lines      : number of dots that contributed valid lines
        """
        def _line_residuals(params, pts):
            p0  = params[:3]
            v   = self.angles_to_unit_vector(params[3], params[4])
            return np.concatenate([np.cross(v, p - p0) for p in pts])

        def _fit_line_3d(pts):
            p0_init = np.mean(pts, axis=0)
            v_init  = pts[-1] - pts[0]
            if np.linalg.norm(v_init) < 1e-9:
                v_init = np.array([1.0, 0.0, 0.0])
            th0, ph0 = self.unit_vector_to_angles(v_init)
            x0 = np.array([*p0_init, th0, ph0], dtype=float)
            lb = np.array([-np.inf, -np.inf, -np.inf, -np.pi / 2, -np.pi])
            ub = np.array([ np.inf,  np.inf,  np.inf,  np.pi / 2,  np.pi])
            res = least_squares(_line_residuals, x0, args=(pts,), bounds=(lb, ub))
            p0 = res.x[:3]
            v  = self.angles_to_unit_vector(res.x[3], res.x[4])
            v /= (np.linalg.norm(v) + 1e-12)
            return p0, v

        I     = np.eye(3)
        A_mat = np.zeros((3, 3))
        b_vec = np.zeros(3)
        n_lines = 0

        for i in range(U_tx.shape[0]):
            pts   = U_tx[i, :, :]
            valid = np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts, axis=1) > 1e-9)
            pts   = pts[valid]
            if pts.shape[0] < 2:
                continue
            p0, v   = _fit_line_3d(pts)
            P        = I - np.outer(v, v)
            A_mat   += P
            b_vec   += P @ p0
            n_lines += 1

        intersection, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return intersection, n_lines

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 5 – Unit vector estimation
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_unit_vectors(self, U: np.ndarray, AB: float) -> np.ndarray:
        """
        Estimate a unit direction vector V[i] for each projected dot.

        For each dot i, Q_ij = U_ij − B are the calibration points expressed
        relative to the transmitter.  V[i] is fitted by minimising cross-product
        residuals (angle error).

        Parameters
        ----------
        U  : (n_dots, n_dist, 3) calibration 3-D points in camera coordinates
        AB : scalar baseline distance (x-offset of transmitter)

        Returns
        -------
        V : (n_dots, 3) unit direction vectors per dot in camera / transmitter space
        """
        B = np.array([AB, 0.0, 0.0], dtype=float)

        def _v_residuals(params, Q):
            v = self.angles_to_unit_vector(params[0], params[1])
            return np.concatenate([np.cross(q, v) for q in Q])

        def _fit_unit_vector(Q):
            m = np.mean(Q, axis=0)
            if np.linalg.norm(m) < 1e-9:
                m = np.array([1.0, 0.0, 0.0])
            th0, ph0 = self.unit_vector_to_angles(m)
            res = least_squares(_v_residuals, [th0, ph0], args=(Q,),
                                bounds=([-np.pi / 2, -np.pi], [np.pi / 2, np.pi]))
            v = self.angles_to_unit_vector(res.x[0], res.x[1])
            v /= (np.linalg.norm(v) + 1e-12)
            return v

        n_dots = U.shape[0]
        V = np.full((n_dots, 3), np.nan, dtype=float)

        for i in range(n_dots):
            pts   = U[i, :, :]
            valid = np.all(np.isfinite(pts), axis=1)
            pts   = pts[valid]
            if pts.shape[0] < 2:
                continue
            Q       = pts - B[None, :]
            V[i, :] = _fit_unit_vector(Q)

        return V

    # ─────────────────────────────────────────────────────────────────────────
    # Runtime helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def cam_ray(u: float, v: float, K_inv: np.ndarray) -> np.ndarray:
        """Return the unit camera ray for pixel (u, v) via the inverse intrinsics."""
        r = K_inv @ np.array([u, v, 1.0], dtype=float)
        r /= (np.linalg.norm(r) + 1e-12)
        return r

    @staticmethod
    def triangulate_depth(u: float, v: float, v_i: np.ndarray,
                          B: np.ndarray, K_inv: np.ndarray) -> float:
        """
        Compute triangulated axial depth Z_tri.

        Solves  s·r − t·v_i = B  in LS (r = camera ray at (u,v), v_i = dot unit vector).
        Returns the axial component: Z_tri = (s·r)[2].

        From Microsoft-Paper eq. for the consistency error.
        """
        r   = DotCalibration.cam_ray(u, v, K_inv)
        v_i = v_i / (np.linalg.norm(v_i) + 1e-12)
        A   = np.column_stack([r, -v_i])   # 3×2
        st, *_ = np.linalg.lstsq(A, B, rcond=None)
        return abs(float(st[0] * r[2]))

    @staticmethod
    def consistency_error(Z_tof: float, Z_tri: float) -> float:
        """
        Compute the Microsoft-Paper consistency error:  ε = |1/Z_ToF − 1/Z_tri|.

        A small ε indicates agreement between iToF and triangulation (MPI-free).
        """
        if Z_tof <= 1e-9 or Z_tri <= 1e-9:
            return np.inf
        return abs(1.0 / Z_tof - 1.0 / Z_tri)
    


    @staticmethod
    def build_calibration_trails(subpixel_list: List[List[dict]],
                                  n_dots: int = 100) -> np.ndarray:
        """
        Build a (n_dist, n_dots, 2) array of pixel-space calibration trails.

        Each trail[j, i, :] = (u, v) of dot i at calibration distance index j.
        Missing entries are filled by linear interpolation.

        Parameters
        ----------
        subpixel_list : list (one entry per distance) of subpixel dicts {id, x, y}
        n_dots        : total number of dots (e.g. 100 for a 10×10 grid)
        """
        n_dist   = len(subpixel_list)
        trail_xy = np.full((n_dist, n_dots, 2), np.nan, dtype=float)

        for j, spx in enumerate(subpixel_list):
            for d in spx:
                i = int(d["id"])
                trail_xy[j, i, 0] = float(d["x"])
                trail_xy[j, i, 1] = float(d["y"])

        jj = np.arange(n_dist, dtype=float)
        for i in range(n_dots):
            xs, ys = trail_xy[:, i, 0], trail_xy[:, i, 1]
            m = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(m):
                continue
            trail_xy[:, i, 0] = np.interp(jj, jj[m], xs[m])
            trail_xy[:, i, 1] = np.interp(jj, jj[m], ys[m])

        return trail_xy
