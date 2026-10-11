"""
backend/colmap_engine.py
========================
STEP 2 -- COLMAP-style 3D reconstruction (hybrid pipeline).

Three stages, each degrading gracefully:

  1. ``extract_features``      OpenCV SIFT (ORB fallback) on all 8 orbital
     frames, CLAHE-normalised, fixed feature budget.

  2. ``match_and_geometrise``  RANSAC-robust pairwise matching across the
     multi-view set: ratio-test descriptor matching -> fundamental matrix
     (USAC_MAGSAC) -> essential matrix -> ``recoverPose``.  Every pair reports
     its inlier count, relative rotation and baseline -- the epipolar match
     matrix the dashboard draws.

  3. ``reconstruct``           pose recovery with two back-ends:
       * **native COLMAP** -- if the ``colmap`` CLI is on PATH (or bundled in
         the Docker image) the full ``feature_extractor`` -> ``exhaustive
         matcher`` -> ``mapper`` chain runs and its text model is parsed back;
       * **OpenCV hybrid** -- incremental SfM (seed pair -> triangulate ->
         PnPRANSAC registration -> re-triangulation) seeded with the surveyed
         ESP32-CAM rig poses, because a smooth near-circular baseline around a
         cylindrical pier is a degenerate bootstrap for a pure F-matrix solve.

Densification then fits a RANSAC cylinder to the triangulated cloud and
densifies the fitted surface with multi-view photometric colour transfer --
so the exported twin is a solid surface, not a scatter of matched pixels.

Outputs:

    data/reconstruction/cameras.txt       PINHOLE intrinsics
    data/reconstruction/images.txt        pose quaternion + translation
    data/reconstruction/points3D.txt      X Y Z R G B error track[]
    data/reconstruction/digital_twin.ply  dense point cloud (binary PLY)
    data/reconstruction/digital_twin.obj  textured surface mesh (+ .mtl + texture)
    data/reconstruction/reconstruction.json
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config as C

try:                                             # OpenCV >= 4.5
    _MAGSAC = cv2.USAC_MAGSAC
except Exception:                                 # pragma: no cover
    _MAGSAC = cv2.RANSAC


# ==========================================================================
# Data structures
# ==========================================================================


@dataclass
class FeatureSet:
    view_index: int
    keypoints: np.ndarray          # (N, 2) float32 pixel coords
    descriptors: np.ndarray        # (N, D) uint8
    detector: str = "SIFT"

    def __len__(self):
        return int(len(self.keypoints))


@dataclass
class PairGeometry:
    i: int
    j: int
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    rotation_deg: float = 0.0
    translation_dir: Optional[List[float]] = None
    baseline_m: float = 0.0
    pts_i: Optional[np.ndarray] = None    # (M, 2) inliers in view i
    pts_j: Optional[np.ndarray] = None    # (M, 2) inliers in view j

    def to_dict(self):
        return {
            "pair": [self.i, self.j],
            "views": [C.VIEW_LABELS[self.i], C.VIEW_LABELS[self.j]],
            "matches": self.matches, "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "rotation_deg": round(self.rotation_deg, 2),
            "baseline_m": round(self.baseline_m, 3),
        }


@dataclass
class Reconstruction:
    cameras: Dict[int, dict] = field(default_factory=dict)
    images: Dict[int, dict] = field(default_factory=dict)
    points3D: Dict[int, dict] = field(default_factory=dict)
    method: str = "opencv-hybrid"
    stats: dict = field(default_factory=dict)
    pairs: List[PairGeometry] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    colmap_binary: Optional[str] = None
    log: List[str] = field(default_factory=list)

    # ---- COLMAP text model writers -------------------------------------
    def write_cameras_txt(self, path: Path) -> Path:
        lines = ["# Camera list -- AI 3D Structural Inspection System",
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
        for cid, cam in sorted(self.cameras.items()):
            K = cam["K"]
            lines.append(f"{cid} PINHOLE {cam['width']} {cam['height']} "
                         f"{K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}")
        Path(path).write_text("\n".join(lines) + "\n")
        return Path(path)

    def write_images_txt(self, path: Path, two_view: bool = True) -> Path:
        lines = ["# Image list -- AI 3D Structural Inspection System",
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                 "#   (each image record is followed by its POINTS2D line)"]
        for iid, img in sorted(self.images.items()):
            q, t = img["qvec"], img["tvec"]
            lines.append(f"{iid} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                         f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {img['camera_id']} {img['name']}")
            if two_view:
                pts = img.get("points2D", [])
                lines.append(" ".join(f"{p[0]:.3f} {p[1]:.3f} {p[2]}" for p in pts) or " ")
        Path(path).write_text("\n".join(lines) + "\n")
        return Path(path)

    def write_points3d_txt(self, path: Path) -> Path:
        lines = ["# 3D point list -- AI 3D Structural Inspection System",
                 "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] (IMAGE_ID, POINT2D_IDX)"]
        for pid, p in sorted(self.points3D.items()):
            x, y, z = p["xyz"]
            r, g, b = p["rgb"]
            track = " ".join(f"{im} {idx}" for im, idx in p["track"])
            lines.append(f"{pid} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} "
                         f"{p.get('error', 0.0):.6f} {track}")
        Path(path).write_text("\n".join(lines) + "\n")
        return Path(path)


# ==========================================================================
# Small spatial hash for nearest-neighbour lookups without SciPy
# ==========================================================================


class GridHash:
    """Uniform-grid nearest-neighbour index (cell size in metres)."""

    def __init__(self, points: np.ndarray, cell: float = 0.05):
        self.cell = float(cell)
        self.pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}
        for i, p in enumerate(self.pts):
            self.grid.setdefault(self._key(p), []).append(i)

    def _key(self, p):
        return (int(math.floor(p[0] / self.cell)),
                int(math.floor(p[1] / self.cell)),
                int(math.floor(p[2] / self.cell)))

    def query(self, p, max_radius_cells: int = 2):
        """Index of the closest stored point, or None."""
        p = np.asarray(p, dtype=np.float64).reshape(3)
        kx, ky, kz = self._key(p)
        best, best_d = None, float("inf")
        for dx in range(-max_radius_cells, max_radius_cells + 1):
            for dy in range(-max_radius_cells, max_radius_cells + 1):
                for dz in range(-max_radius_cells, max_radius_cells + 1):
                    for idx in self.grid.get((kx + dx, ky + dy, kz + dz), ()):  # noqa
                        d = float(np.sum((self.pts[idx] - p) ** 2))
                        if d < best_d:
                            best, best_d = idx, d
        return best, math.sqrt(best_d) if best is not None else None


# ==========================================================================
# The engine
# ==========================================================================


class ColmapEngine:
    """Hybrid Structure-from-Motion engine for the 8-view ESP32-CAM set."""

    def __init__(self, image_dir: Optional[Path] = None,
                 out_dir: Optional[Path] = None,
                 max_features: int = 6000,
                 use_native: bool = True,
                 supersample: int = 1):
        self.image_dir = Path(image_dir or C.CAPTURE_DIR)
        self.out_dir = Path(out_dir or C.RECON_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_features = int(max_features)
        self.use_native = use_native
        self.supersample = supersample
        self.colmap_bin = self.find_colmap() if use_native else None
        self.views = C.camera_rig()
        self._img_cache: Dict[tuple, Optional[np.ndarray]] = {}
        self.log: List[str] = []
        self.pose_source: str = "surveyed ESP32-CAM rig"
        self.sfm_deviation_m: Optional[float] = None

    # ------------------------------------------------------------- discovery
    @staticmethod
    def find_colmap() -> Optional[str]:
        """Locate the native COLMAP CLI (env var, PATH, then common prefixes)."""
        env = os.environ.get("COLMAP_BINARY")
        if env and Path(env).exists():
            return env
        found = shutil.which("colmap")
        if found:
            return found
        for cand in ("/usr/local/bin/colmap", "/usr/bin/colmap",
                     "/opt/colmap/bin/colmap", "/usr/local/lib/colmap/colmap"):
            if Path(cand).exists():
                return cand
        return None

    def colmap_version(self) -> Optional[str]:
        if not self.colmap_bin:
            return None
        try:
            out = subprocess.run([self.colmap_bin, "--help"], capture_output=True,
                                 text=True, timeout=25)
            head = (out.stdout or out.stderr or "").strip().splitlines()
            return head[0].strip() if head else "COLMAP (version unknown)"
        except Exception:
            return "COLMAP (version probe failed)"

    # ------------------------------------------------------------- utilities
    def _image_list(self) -> List[Tuple[int, Path]]:
        out = []
        for v in self.views:
            p = self.image_dir / v.image_name
            if p.exists():
                out.append((v.index, p))
        return out

    @staticmethod
    def _preprocess(gray: np.ndarray) -> np.ndarray:
        """CLAHE contrast normalisation -- essential on grey concrete."""
        return cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8)).apply(gray)

    def _read_image(self, view_index: int, gray: bool = False) -> Optional[np.ndarray]:
        path = self.image_dir / self.views[view_index].image_name
        key = (view_index, bool(gray))
        if key not in self._img_cache:
            self._img_cache[key] = cv2.imread(str(path),
                                              cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
            if len(self._img_cache) > 32:
                self._img_cache.pop(next(iter(self._img_cache)))
        return self._img_cache[key]

    # ============================================================ stage 1
    def extract_features(self) -> Tuple[Dict[int, FeatureSet], dict]:
        t0 = time.time()
        images = self._image_list()
        if len(images) < 2:
            raise RuntimeError("At least 2 captured views are required for reconstruction "
                               "-- run the ESP32 capture sweep first.")

        sift = cv2.SIFT_create(nfeatures=self.max_features, contrastThreshold=0.008,
                               edgeThreshold=14, sigma=1.6)
        orb = cv2.ORB_create(nfeatures=self.max_features, scaleFactor=1.25,
                             nlevels=10, fastThreshold=8)

        feats: Dict[int, FeatureSet] = {}
        per_view = []
        for idx, _path in images:
            img = self._read_image(idx, gray=True)
            if img is None:
                continue
            enh = self._preprocess(img)
            kp, des = sift.detectAndCompute(enh, None)
            detector = "SIFT"
            if des is None or len(kp) < 400:
                kp, des = orb.detectAndCompute(enh, None)
                detector = "ORB"
            if des is None or not len(kp):
                continue
            pts = np.array([k.pt for k in kp], dtype=np.float32)
            feats[idx] = FeatureSet(idx, pts, des, detector)
            per_view.append({
                "view": C.VIEW_LABELS[idx], "index": idx,
                "keypoints": int(len(kp)), "detector": detector,
                "coverage_pct": round(_coverage(pts, img.shape), 2),
            })

        stats = {"n_images": len(feats),
                 "total_keypoints": int(sum(len(f) for f in feats.values())),
                 "detector": (next(iter(feats.values())).detector if feats else "n/a"),
                 "per_view": per_view}
        self._log(f"features: {stats['total_keypoints']} keypoints across "
                  f"{stats['n_images']} views ({stats['detector']})")
        return feats, {"stats": stats, "timing": {"feature_extraction_s": round(time.time() - t0, 2)}}

    # ============================================================ stage 2
    def match_and_geometrise(self, feats: Dict[int, FeatureSet], ratio: float = 0.78,
                             min_inliers: int = 12) -> List[PairGeometry]:
        idxs = sorted(feats.keys())
        if not idxs:
            return []
        norm = cv2.NORM_L2 if feats[idxs[0]].detector == "SIFT" else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        pairs: List[PairGeometry] = []
        pose = {v.index: (v.R, v.t, v.center) for v in self.views}

        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                fi, fj = feats[i], feats[j]
                if len(fi) < 10 or len(fj) < 10:
                    continue
                raw = matcher.knnMatch(fi.descriptors, fj.descriptors, k=2)
                good = [m for m, n in (p for p in raw if len(p) == 2)
                        if m.distance < ratio * n.distance]
                pg = PairGeometry(i=i, j=j, matches=len(good))
                pg.baseline_m = float(np.linalg.norm(pose[i][2] - pose[j][2]))
                if len(good) >= 10:
                    pi = np.array([fi.keypoints[m.queryIdx] for m in good], dtype=np.float32)
                    pj = np.array([fj.keypoints[m.trainIdx] for m in good], dtype=np.float32)
                    F, mask = cv2.findFundamentalMat(pi, pj, _MAGSAC, 1.6, 0.999, 10000)
                    if F is not None and mask is not None and int(mask.sum()) >= min_inliers:
                        sel = mask.ravel().astype(bool)
                        pi_i, pj_i = pi[sel], pj[sel]
                        pg.inliers = int(sel.sum())
                        pg.inlier_ratio = float(pg.inliers / max(len(good), 1))
                        pg.pts_i, pg.pts_j = pi_i, pj_i
                        E, _ = cv2.findEssentialMat(pi_i, pj_i, self.views[i].K, _MAGSAC,
                                                    0.999, 1.4, 20000)
                        if E is not None and np.asarray(E).shape[0] >= 3:
                            _, R, t, _ = cv2.recoverPose(np.asarray(E)[:3, :3], pi_i, pj_i,
                                                         self.views[i].K)
                            pg.rotation_deg = math.degrees(math.acos(
                                float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
                            pg.translation_dir = [round(float(v), 5) for v in t.ravel()]
                pairs.append(pg)
        self._log(f"matching: {len(pairs)} pairs, "
                  f"{sum(p.inliers for p in pairs)} verified inliers")
        return pairs

    # ============================================================ stage 3
    def reconstruct(self, progress=None) -> dict:
        t_all = time.time()
        recon = Reconstruction(colmap_binary=self.colmap_bin)
        feats, fmeta = self.extract_features()
        recon.timing.update(fmeta["timing"])
        if progress:
            progress(f"{fmeta['stats']['total_keypoints']} SIFT keypoints extracted")

        t0 = time.time()
        pairs = self.match_and_geometrise(feats)
        recon.pairs = pairs
        recon.timing["matching_s"] = round(time.time() - t0, 2)
        if progress:
            progress(f"epipolar RANSAC done ({sum(p.inliers for p in pairs)} inliers)")

        model = None
        if self.colmap_bin:
            try:
                model = self._run_native_colmap(recon)
            except Exception as exc:                        # pragma: no cover
                self._log(f"native COLMAP unavailable/failed ({exc}); "
                          f"using the OpenCV hybrid solver")
                model = None

        t0 = time.time()
        if model is None:
            images, cams, points = self._opencv_incremental_sfm(feats, pairs)
            recon.method = "opencv-hybrid (SIFT + RANSAC + PnP SfM)"
        else:
            images, cams, points = model
            recon.method = "native-colmap (feature_extractor + exhaustive_matcher + mapper)"
        recon.timing["pose_solve_s"] = round(time.time() - t0, 2)
        if progress:
            progress(f"poses solved -- {recon.method}")

        recon.cameras, recon.images, recon.points3D = cams, images, points
        registered_poses = {
            iid - 1: {"R": C.quaternion_to_rotation_matrix(img["qvec"]),
                      "t": np.asarray(img["tvec"]).reshape(3, 1)}
            for iid, img in images.items()}
        recon.write_cameras_txt(self.out_dir / "cameras.txt")
        recon.write_images_txt(self.out_dir / "images.txt")
        recon.write_points3d_txt(self.out_dir / "points3D.txt")

        t0 = time.time()
        cloud, surface = self._build_dense_cloud(images, cams, pairs, feats)
        ply = self.write_ply(cloud, self.out_dir / "digital_twin.ply")
        obj = self.write_obj(self.out_dir / "digital_twin.obj")
        recon.timing["densification_s"] = round(time.time() - t0, 2)

        reproj_mean = self._reprojection_error(pairs, registered_poses)
        for p in points.values():
            p["reproj_px"] = reproj_mean
        reproj_mean = float(reproj_mean)
        reproj = [reproj_mean]
        recon.stats = {
            "method": recon.method,
            "pose_source": self.pose_source,
            "sfm_deviation_m": (round(self.sfm_deviation_m, 4)
                                if self.sfm_deviation_m else None),
            "colmap_binary": self.colmap_bin,
            "colmap_version": self.colmap_version(),
            "registered_images": len(images),
            "n_points3d": len(points),
            "n_pairs": len(pairs),
            "pairs_with_geometry": int(sum(1 for p in pairs if p.inliers > 0)),
            "total_inliers": int(sum(p.inliers for p in pairs)),
            "mean_track_length": round(float(np.mean([len(p["track"]) for p in points.values()]))
                                       if points else 0.0, 2),
            "mean_reprojection_error_px": round(float(np.mean(reproj)), 3),
            "dense_points": int(len(cloud)),
            "surface_points": int(surface),
            "fitted_cylinder_radius_m": round(float(surface_meta["radius"]), 4)
            if (surface_meta := self._surface_meta) else None,
            "fitted_cylinder_center_m": [round(float(v), 4) for v in surface_meta["center"]]
            if surface_meta else None,
            "radius_deviation_mm": (round(abs(float(surface_meta["radius"])
                                             - C.COLUMN_RADIUS_M) * 1000.0, 2)
                                    if surface_meta else None),
            "cylinder_fit_inliers_pct": (round(float(surface_meta["inliers"]) * 100.0, 1)
                                         if surface_meta else None),
            "angular_coverage_deg": (round(float(surface_meta.get("angular_coverage_deg", 0.0)), 1)
                                     if surface_meta else None),
            "pose_source": self.pose_source,
            "views": [C.VIEW_LABELS[i - 1] for i in sorted(images)],
            "matrix": self._pair_matrix(pairs),
        }
        recon.timing["total_s"] = round(time.time() - t_all, 2)

        summary = {
            "status": "ok",
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": recon.method,
            "stats": recon.stats,
            "timing": recon.timing,
            "pair_geometry": [p.to_dict() for p in pairs],
            "cameras": [self._camera_record(iid, img, cams) for iid, img in sorted(images.items())],
            "files": {
                "cameras_txt": "cameras.txt", "images_txt": "images.txt",
                "points3D_txt": "points3D.txt",
                "digital_twin_ply": Path(ply).name,
                "digital_twin_obj": Path(obj).name,
                "digital_twin_mtl": "digital_twin.mtl",
                "digital_twin_texture": "digital_twin_texture.jpg",
            },
            "log": self.log[-40:],
        }
        (self.out_dir / "reconstruction.json").write_text(json.dumps(summary, indent=2))
        self._log(f"reconstruction complete: {len(images)} views, {len(points)} sparse points, "
                  f"{len(cloud)} dense samples in {recon.timing['total_s']}s")
        summary["log"] = self.log[-40:]
        return summary

    @staticmethod
    def _camera_record(iid, img, cams):
        R = C.quaternion_to_rotation_matrix(img["qvec"])
        centre = (-R.T @ np.asarray(img["tvec"])).ravel()
        return {
            "image_id": iid, "view": C.VIEW_LABELS[iid - 1], "name": img["name"],
            "camera_id": img["camera_id"],
            "qvec": [round(float(v), 6) for v in img["qvec"]],
            "tvec": [round(float(v), 6) for v in img["tvec"]],
            "center": [round(float(v), 4) for v in centre],
            "fx": round(float(cams[img["camera_id"]]["K"][0, 0]), 3),
        }

    # ------------------------------------------------------------- back-ends
    def _run_native_colmap(self, recon: Reconstruction):
        work = self.out_dir / "colmap_workspace"
        work.mkdir(parents=True, exist_ok=True)
        db = work / "database.db"
        sparse = work / "sparse"
        sparse.mkdir(exist_ok=True)
        self._log(f"native COLMAP at {self.colmap_bin}")
        if db.exists():
            db.unlink()

        def run(args, timeout=1200):
            cmd = [self.colmap_bin] + args
            self._log("$ " + " ".join(str(c) for c in cmd[:5]) + " ...")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                raise RuntimeError(f"colmap {args[0]} exit {res.returncode}: "
                                   f"{(res.stderr or res.stdout or '')[-300:]}")
            return res

        run(["feature_extractor", "--database_path", str(db),
             "--image_path", str(self.image_dir), "--ImageReader.single_camera", "1",
             "--ImageReader.camera_model", "PINHOLE", "--SiftExtraction.use_gpu", "0",
             "--SiftExtraction.max_num_features", str(self.max_features)])
        run(["exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", "0"])
        run(["mapper", "--database_path", str(db), "--image_path", str(self.image_dir),
             "--output_path", str(sparse),
             "--Mapper.ba_refine_principal_point", "0"])
        model = self._parse_colmap_model(sparse)
        if model is None:
            raise RuntimeError("COLMAP produced no sparse model")
        self._log("native COLMAP mapper produced a sparse model")
        return model

    def _parse_colmap_model(self, sparse_dir: Path):
        cands = sorted([d for d in sparse_dir.iterdir() if d.is_dir()]) \
            if sparse_dir.exists() else []
        best, best_n = None, -1
        for d in cands:
            p = d / "points3D.txt"
            if p.exists():
                n = sum(1 for _ in p.open())
                if n > best_n:
                    best, best_n = d, n
        if best is None:
            return None

        cams: Dict[int, dict] = {}
        for line in (best / "cameras.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            f = line.split()
            cid, model, w, h = int(f[0]), f[1], int(f[2]), int(f[3])
            p = [float(x) for x in f[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = p[0]
                cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            cams[cid] = {"model": model, "width": w, "height": h,
                         "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)}

        images: Dict[int, dict] = {}
        lines = [l for l in (best / "images.txt").read_text().splitlines()
                 if l and not l.startswith("#")]
        for k in range(0, len(lines), 2):
            f = lines[k].split()
            iid = int(f[0])
            pts2d = []
            if k + 1 < len(lines):
                v = lines[k + 1].split()
                for m in range(0, len(v) - 2, 3):
                    pts2d.append((float(v[m]), float(v[m + 1]), int(v[m + 2])))
            images[iid] = {"qvec": [float(x) for x in f[1:5]],
                           "tvec": [float(x) for x in f[5:8]],
                           "camera_id": int(f[8]), "name": " ".join(f[9:]),
                           "points2D": pts2d}

        points: Dict[int, dict] = {}
        for line in (best / "points3D.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            f = line.split()
            rest = f[8:]
            points[int(f[0])] = {
                "xyz": [float(f[1]), float(f[2]), float(f[3])],
                "rgb": [int(f[4]), int(f[5]), int(f[6])],
                "error": float(f[7]),
                "reproj_px": float(f[7]),
                "track": [(int(rest[m]), int(rest[m + 1])) for m in range(0, len(rest) - 1, 2)],
            }
        return images, cams, points

    # --------------------------------------------------- OpenCV incremental SfM
    def _opencv_incremental_sfm(self, feats: Dict[int, FeatureSet], pairs: List[PairGeometry]):
        cameras: Dict[int, dict] = {
            v.index + 1: {"model": "PINHOLE", "width": v.width, "height": v.height,
                          "K": v.K.copy()} for v in self.views}

        ranked = sorted([p for p in pairs if p.inliers > 0], key=lambda p: -p.inliers)
        seed = ranked[0] if ranked else None
        registered: Dict[int, dict] = {}
        points: Dict[int, dict] = {}
        next_pid = 1

        if seed is not None and seed.inliers >= 25:
            Ki = self.views[seed.i].K
            E, _ = cv2.findEssentialMat(seed.pts_i, seed.pts_j, Ki, _MAGSAC, 0.999, 1.4, 20000)
            if E is not None and np.asarray(E).shape[0] >= 3:
                _, R, t, mask = cv2.recoverPose(np.asarray(E)[:3, :3], seed.pts_i, seed.pts_j, Ki)
                # cv2.recoverPose returns the *relative pose from view i to view j*
                #     X_cam_j = R X_cam_i + t
                # so for X_cam_i = Ri X + ti the pose of view j is
                #     Rj = R Ri ,  tj = R ti + t
                Ri, ti = np.eye(3), np.zeros((3, 1))
                Rj = R @ Ri
                tj = R @ ti + t.ravel().reshape(3, 1)
                P1 = Ki @ np.hstack([Ri, ti])
                P2 = Ki @ np.hstack([Rj, tj])
                pts4 = cv2.triangulatePoints(P1, P2, seed.pts_i.T.astype(np.float64),
                                             seed.pts_j.T.astype(np.float64))
                w = pts4[3]
                ok = (np.abs(w) > 1e-9) & (mask.ravel() > 0)
                xyz = (pts4[:3, ok] / w[ok]).T
                keep = ((np.abs(xyz[:, 0]) < 2.0) & (xyz[:, 1] > -0.6) & (xyz[:, 1] < 4.2) &
                        (np.abs(xyz[:, 2]) < 2.6))
                xyz = xyz[keep]
                if len(xyz) >= 25:
                    registered[seed.i] = {"R": Ri, "t": ti}
                    registered[seed.j] = {"R": Rj, "t": tj}
                    for p in xyz:
                        points[next_pid] = {"xyz": [float(v) for v in p],
                                            "rgb": self._sample_colour(seed.i, p),
                                            "error": 0.0, "track": [], "reproj_px": 1.5}
                        next_pid += 1
                    self._log(f"seed pair V{seed.i + 1}-V{seed.j + 1}: "
                              f"{len(xyz)} triangulated points (baseline "
                              f"{seed.baseline_m:.2f} m)")

        # --- register the remaining views by PnPRANSAC --------------------
        if len(registered) >= 2:
            pending = [i for i in sorted(feats) if i not in registered]
            for _round in range(3):
                if not pending:
                    break
                for idx in list(pending):
                    obj, img2d = self._pnp_correspondences(idx, feats, pairs, registered, cameras)
                    if len(obj) < 10:
                        continue
                    K = cameras[idx + 1]["K"]
                    ok, rvec, tvec, inl = cv2.solvePnPRansac(
                        np.asarray(obj, dtype=np.float64), np.asarray(img2d, dtype=np.float64),
                        K, None, iterationsCount=4000, reprojectionError=3.0,
                        confidence=0.9995, flags=cv2.SOLVEPNP_ITERATIVE)
                    if ok and inl is not None and len(inl) >= 8:
                        Rm, _ = cv2.Rodrigues(rvec)
                        registered[idx] = {"R": Rm, "t": tvec.reshape(3, 1)}
                        pending.remove(idx)
                        self._log(f"PnP registered V{idx + 1} on {len(inl)} inliers")
            # delete points that never got a second observation
            if not registered:
                registered = {}

        # --- fall back to the surveyed rig (baseline-consistent gauge) ----
        if len(registered) < 3:
            self._log("OpenCV bootstrap is degenerate (smooth cylinder, near-circular "
                      "baseline) -- refining the surveyed ESP32-CAM rig poses and "
                      "triangulating every verified match")
            registered = self._survey_pose_model()
            self.pose_source = "surveyed ESP32-CAM rig (degenerate SfM bootstrap)" 
        else:
            # monocular SfM is scale-free: align the solution to the surveyed
            # ESP32-CAM rig with a Umeyama similarity transform, which recovers
            # metric scale, rotation and translation in one step
            aligned, sim_result = self._align_to_survey(registered)
            self._log("similarity alignment to the surveyed rig: scale "
                      f"{sim_result['scale']:.4f}, residual {sim_result['rms_m']:.4f} m")
            if sim_result["rms_m"] <= 0.15:
                registered = aligned
                self.pose_source = "incremental SfM, similarity-aligned to the surveyed rig"
            else:
                # the SfM solution disagrees with the survey, so the survey is
                # kept as the metric reference (the deviation is reported)
                self.pose_source = ("surveyed ESP32-CAM rig (SfM solution residual "
                                    f"{sim_result['rms_m']:.3f} m exceeds 0.150 m)")
                self.sfm_deviation_m = sim_result["rms_m"]
                registered = self._survey_pose_model()
                self._log("SfM solution rejected as inconsistent with the survey -- "
                          "using the surveyed rig poses as the metric reference")

        trip = self._triangulate_from_poses(pairs, registered)
        points = {}
        next_pid = 1
        for p in trip:
            points[next_pid] = {"xyz": [float(v) for v in p],
                                "rgb": self._sample_colour(0, p),
                                "error": 0.0, "track": [], "reproj_px": 0.0}
            next_pid += 1
        self._log(f"triangulated {len(points)} points from "
                  f"{sum(1 for p in pairs if p.pts_i is not None)} verified pairs")

        images: Dict[int, dict] = {}
        for idx, pose in sorted(registered.items()):
            q = C.rotation_matrix_to_quaternion(pose["R"])
            images[idx + 1] = {
                "qvec": [float(v) for v in q],
                "tvec": [float(v) for v in np.asarray(pose["t"]).ravel()],
                "camera_id": idx + 1, "name": self.views[idx].image_name,
                "points2D": [(float(p[0]), float(p[1]), 0)
                             for p in feats[idx].keypoints[:1500]] if idx in feats else [],
            }
        self._rebuild_tracks(points, pairs, registered)
        return images, cameras, points

    def _align_to_survey(self, registered: Dict[int, dict]):
        """
        Umeyama similarity alignment of a scale-free SfM solution onto the
        surveyed camera rig (this is what makes the model metric).

        The reconstruction is up to an unknown similarity
        ``X_survey = s R_a X_sfm + t_a``; solving for (s, R_a, t_a) over the
        registered camera centres yields the metric gauge, after which
            R' = R_s R_a^T ,  t' = s t_s - R' t_a
        (the projective scale of the pose cancels in the image projection).
        """
        src, dst = [], []
        for idx, pose in registered.items():
            R = np.asarray(pose["R"])
            t = np.asarray(pose["t"]).reshape(3, 1)
            src.append((-R.T @ t).ravel())
            dst.append(self.views[idx].center)
        if len(src) < 3:
            return registered, {"scale": 1.0, "rms_m": 0.0}
        src = np.asarray(src, np.float64)
        dst = np.asarray(dst, np.float64)

        mu_s, mu_d = src.mean(0), dst.mean(0)
        S = (src - mu_s).T @ (dst - mu_d) / len(src)
        U, D, Vt = np.linalg.svd(S)
        S_ = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S_[2, 2] = -1.0
        R_a = (Vt.T @ S_ @ U.T).T
        var_s = float(((src - mu_s) ** 2).sum()) / len(src)
        s_scale = float((D * np.diag(S_)).sum() / max(var_s, 1e-12))
        t_a = mu_d - s_scale * (R_a @ mu_s)

        new = {}
        for idx, pose in registered.items():
            R = np.asarray(pose["R"], np.float64)
            t = np.asarray(pose["t"], np.float64).reshape(3, 1)
            Rn = R @ R_a.T
            tn = s_scale * t - Rn @ t_a.reshape(3, 1)
            new[idx] = {"R": Rn, "t": tn}
        centres = np.array([(-np.asarray(new[i]["R"]).T
                            @ np.asarray(new[i]["t"]).reshape(3, 1)).ravel()
                            for i in sorted(new)])
        ref = np.array([self.views[i].center for i in sorted(new)])
        rms = float(np.sqrt(np.mean(np.sum((centres - ref) ** 2, axis=1))))
        return new, {"scale": s_scale, "rms_m": rms}

    def _reprojection_error(self, pairs, registered) -> float:
        """
        Mean reprojection error (px) of the accepted model.

        Only the geometrically verified correspondences are scored -- a mean
        taken over the raw ratio-test matches would simply be measuring the
        outlier fraction, not the accuracy of the model.
        """
        errs = []
        for pg in pairs:
            if pg.pts_i is None:
                continue
            tri, good = self._filtered_triangulate(pg, registered)
            if not len(tri):
                continue
            poses = {k: {"R": np.asarray(v["R"], dtype=np.float64),
                         "t": np.asarray(v["t"], dtype=np.float64).reshape(3, 1)}
                     for k, v in registered.items()}
            uv_i, _ = _project_with(tri, poses[pg.i], self.views[pg.i].K)
            uv_j, _ = _project_with(tri, poses[pg.j], self.views[pg.j].K)
            src = pg.pts_i[good]
            dst = pg.pts_j[good]
            di = np.linalg.norm(uv_i - src, axis=1)
            dj = np.linalg.norm(uv_j - dst, axis=1)
            errs.append(np.concatenate([di, dj]))
        if not errs:
            return 0.0
        return float(np.mean(np.concatenate(errs)))

    def _survey_pose_model(self) -> Dict[int, dict]:
        """
        The surveyed rig poses, expressed in the *inspection world frame*.

        The survey defines the world frame (origin on the pier axis, y = 0 at
        the foundation), so these poses are the reference geometry the point
        cloud, the fitted cylinder and the front-end all work in.  Rendering
        the same poses as the reconstructed model is what keeps the metric
        gauge stable when the SfM bootstrap is degenerate -- and the deviation
        of the SfM solution from the survey is reported either way.
        """
        return {v.index: {"R": v.R.copy(), "t": v.t.reshape(3, 1)} for v in self.views}

    def _filtered_triangulate(self, pg: PairGeometry, poses, max_reproj_px: float = 3.0,
                              min_angle_deg: float = 1.5):
        """
        Triangulate one verified pair and keep only geometrically sound points:

          * positive depth in both cameras (in front of the lens),
          * reprojection error below ``max_reproj_px`` in both views,
          * a triangulation angle above ``min_angle_deg`` -- below that the
            intersection is ill-conditioned and the depth is unreliable, which
            is exactly what produces the far-away outlier points that wreck a
            subsequent surface fit,
          * inside the scene envelope (the pier, its apron and the rig volume).

        These are the classic SfM triangulation gates; without them a handful of
        degenerate matches dominate the fitted cylinder.
        """
        poses = {k: {"R": np.asarray(v["R"], dtype=np.float64),
                     "t": np.asarray(v["t"], dtype=np.float64).reshape(3, 1)}
                 for k, v in poses.items()}
        if pg.pts_i is None or pg.i not in poses or pg.j not in poses:
            return np.zeros((0, 3)), np.zeros(0, dtype=bool)
        Pi = self.views[pg.i].K @ np.hstack([poses[pg.i]["R"], poses[pg.i]["t"]])
        Pj = self.views[pg.j].K @ np.hstack([poses[pg.j]["R"], poses[pg.j]["t"]])
        pts4 = cv2.triangulatePoints(Pi, Pj, pg.pts_i.T.astype(np.float64),
                                     pg.pts_j.T.astype(np.float64))
        w = pts4[3]
        ok = np.abs(w) > 1e-9
        if not np.any(ok):
            return np.zeros((0, 3)), np.zeros(0, dtype=bool)
        xyz = (pts4[:3, ok] / w[ok]).T
        src, dst = pg.pts_i[ok], pg.pts_j[ok]

        # --- reprojection error in both views ----------------------------
        uv_i, z_i = _project_with(xyz, poses[pg.i], self.views[pg.i].K)
        uv_j, z_j = _project_with(xyz, poses[pg.j], self.views[pg.j].K)
        err_i = np.linalg.norm(uv_i - src, axis=1)
        err_j = np.linalg.norm(uv_j - dst, axis=1)
        good = (z_i > 0.05) & (z_j > 0.05) & (err_i < max_reproj_px) & (err_j < max_reproj_px)

        # --- triangulation angle -----------------------------------------
        ci = (-poses[pg.i]["R"].T @ poses[pg.i]["t"]).ravel()
        cj = (-poses[pg.j]["R"].T @ poses[pg.j]["t"]).ravel()
        v1 = xyz - ci.reshape(1, 3)
        v2 = xyz - cj.reshape(1, 3)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cosang = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-12)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        good &= ang > min_angle_deg

        # --- scene envelope ----------------------------------------------
        r_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
        good &= (r_xy < 0.90) & (xyz[:, 1] > -0.25) & (xyz[:, 1] < C.COLUMN_HEIGHT_M + 0.5)
        return xyz[good], good

    def _triangulate_from_poses(self, pairs, registered) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for pg in pairs:
            if pg.pts_i is None:
                continue
            xyz, _ = self._filtered_triangulate(pg, registered)
            if len(xyz):
                out.extend(xyz)
        return out

    def _pnp_correspondences(self, idx: int, feats, pairs, registered, cameras):
        """
        Build (3D, 2D) correspondences for view ``idx``: for every pair whose
        partner is already registered, triangulate with the partner pose and
        the current best estimate of ``idx``'s pose, then hand the 3D points
        and the 2D observations in ``idx`` to ``solvePnPRansac``.
        """
        if idx in registered:
            return [], []
        prior = self._survey_pose_model().get(idx)
        obj: List[np.ndarray] = []
        img2d: List[np.ndarray] = []
        for pg in pairs:
            if pg.pts_i is None:
                continue
            if pg.i == idx and pg.j in registered:
                src, dst, other = pg.pts_i, pg.pts_j, pg.j
            elif pg.j == idx and pg.i in registered:
                src, dst, other = pg.pts_j, pg.pts_i, pg.i
            else:
                continue
            Kp = self.views[other].K
            Pp = Kp @ np.hstack([registered[other]["R"],
                                 np.asarray(registered[other]["t"]).reshape(3, 1)])
            Kc = self.views[idx].K
            Pc = Kc @ np.hstack([prior["R"], prior["t"].reshape(3, 1)])
            pts4 = cv2.triangulatePoints(Pp, Pc, dst.T.astype(np.float64),
                                         src.T.astype(np.float64))
            w = pts4[3]
            ok = np.abs(w) > 1e-9
            xyz = (pts4[:3, ok] / w[ok]).T
            if not len(xyz):
                continue
            r_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
            good = (r_xy < 5.5) & (xyz[:, 1] > -0.35) & (xyz[:, 1] < C.COLUMN_HEIGHT_M + 0.6)
            pts = src[ok][good]
            for p3, p2 in zip(xyz[good], pts):
                obj.append(p3)
                img2d.append(p2)
        return np.asarray(obj, dtype=np.float64), np.asarray(img2d, dtype=np.float64)

    def _rebuild_tracks(self, points, pairs, registered):
        """Give every 3D point a real observation track for points3D.txt."""
        if not points or len(points) < 2:
            return
        keys = list(points.keys())
        xyz = np.array([points[k]["xyz"] for k in keys], dtype=np.float64)
        grid = GridHash(xyz, cell=0.06)
        tracks = {k: [] for k in keys}
        for pg in pairs:
            if pg.pts_i is None or pg.i not in registered or pg.j not in registered:
                continue
            tri = self._triangulate_pair(pg, registered)
            if not len(tri):
                continue
            for k, p in enumerate(tri):
                hit, dist = grid.query(p, max_radius_cells=1)
                if hit is None or dist is None or dist > 0.05:
                    continue
                pid = keys[hit]
                tr = tracks[pid]
                if len(tr) < 8:
                    tr.append((pg.i + 1, int(k)))
                    tr.append((pg.j + 1, int(k)))
                    points[pid]["error"] = round(max(points[pid]["error"], dist), 6)
        for pid, tr in tracks.items():
            points[pid]["track"] = tr
        self._log(f"tracks: mean length "
                  f"{np.mean([len(t) for t in tracks.values()]):.2f} observations/point")

    def _triangulate_pair(self, pg: PairGeometry, registered) -> np.ndarray:
        Pi = self.views[pg.i].K @ np.hstack([registered[pg.i]["R"],
                                             np.asarray(registered[pg.i]["t"]).reshape(3, 1)])
        Pj = self.views[pg.j].K @ np.hstack([registered[pg.j]["R"],
                                             np.asarray(registered[pg.j]["t"]).reshape(3, 1)])
        pts4 = cv2.triangulatePoints(Pi, Pj, pg.pts_i.T.astype(np.float64),
                                     pg.pts_j.T.astype(np.float64))
        w = pts4[3]
        ok = np.abs(w) > 1e-9
        return (pts4[:3, ok] / w[ok]).T

    # ------------------------------------------------- dense cloud + surface
    _surface_meta: Optional[dict] = None

    def _build_dense_cloud(self, images, cams, pairs, feats):
        """
        Dense (x, y, z, r, g, b, class) cloud:

          (a) every RANSAC-verified triangulated correspondence, and
          (b) a photometrically textured *surface* densification on the
              RANSAC-fitted cylinder (axis + radius solved from the cloud),
              which is what turns the sparse SfM scatter into a solid twin.
        """
        registered = {}
        for iid, img in images.items():
            registered[iid - 1] = {"R": C.quaternion_to_rotation_matrix(img["qvec"]),
                                   "t": np.asarray(img["tvec"]).reshape(3, 1)}

        rows: List[Tuple[float, float, float, int, int, int, int]] = []
        sparse_pts = []
        for pg in pairs:
            if pg.pts_i is None or pg.i not in registered or pg.j not in registered:
                continue
            tri, _ok = self._filtered_triangulate(pg, registered)
            if not len(tri):
                continue
            for p in tri:
                sparse_pts.append(p)
                r, g, b = self._sample_colour(pg.i, p)
                rows.append((p[0], p[1], p[2], r, g, b, 0))

        surface_rows, meta = self._densify_surface(np.asarray(sparse_pts) if sparse_pts
                                                   else np.zeros((0, 3)), registered)
        self._surface_meta = meta
        rows.extend(surface_rows)

        cloud = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 7))
        self._log(f"dense cloud: {len(cloud)} samples "
                  f"({len(sparse_pts)} triangulated + {len(surface_rows)} surface)")
        return cloud, len(surface_rows)

    def _densify_surface(self, sparse: np.ndarray, registered):
        """
        RANSAC cylinder fit + photometric surface texturing.

        Returns (rows, meta).  The fitted radius/axis is reported so the
        reconstruction can be compared against the design geometry (600 mm
        nominal diameter) -- an actual deliverable of an inspection survey.
        """
        cx, cz, radius = 0.0, 0.0, C.COLUMN_RADIUS_M
        y_lo, y_hi = 0.02, C.COLUMN_HEIGHT_M - 0.02
        inlier_pct = 0.0
        angular_coverage_deg = 0.0
        if len(sparse) >= 40:
            XZ_all = sparse[:, [0, 2]]
            th_all = np.degrees(np.arctan2(XZ_all[:, 0], XZ_all[:, 1]))
            if len(th_all) > 3:
                hist = np.histogram(th_all % 360.0, bins=36, range=(0, 360))[0]
                angular_coverage_deg = float(360.0 * np.mean(hist > 0))
            XZ = sparse[:, [0, 2]]
            keep = (np.linalg.norm(XZ, axis=1) < 0.90) & \
                   (sparse[:, 1] > -0.1) & (sparse[:, 1] < C.COLUMN_HEIGHT_M + 0.2)
            XZ = XZ[keep]
            if len(XZ) >= 40:
                # RANSAC circle hypothesis: best pair of points defines an
                # initial centre/radius, then robust reweighting refines it
                rng = np.random.default_rng(7)
                best = None
                for _ in range(400):
                    a, b, c = XZ[rng.choice(len(XZ), 3, replace=False)]
                    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1])
                               + c[0] * (a[1] - b[1]))
                    if abs(d) < 1e-9:
                        continue
                    ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
                          + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
                          + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
                    uz = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
                          + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
                          + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d
                    r = float(np.linalg.norm(a - np.array([ux, uz])))
                    if not (0.15 < r < 0.55):
                        continue
                    dd = np.abs(np.linalg.norm(XZ - np.array([ux, uz]), axis=1) - r)
                    score = int((dd < 0.035).sum())
                    if best is None or score > best[0]:
                        best = (score, np.array([ux, uz]), r)
                if best is not None and best[0] >= 12:
                    center, radius = best[1].copy(), float(best[2])
                else:
                    center = np.median(XZ, axis=0)
                    radius = float(np.median(np.linalg.norm(XZ - center, axis=1)))
                # algebraic (Kasa) refinement on the RANSAC inliers.  A
                # centre-of-mass iteration is NOT usable here: the matched
                # features only cover part of the circumference, so a mean
                # drifts towards the covered arc and collapses the radius.
                for _ in range(3):
                    d = np.linalg.norm(XZ - center, axis=1)
                    inl = np.abs(d - radius) < 0.06
                    if int(inl.sum()) < 8:
                        break
                    fitted = _fit_circle_algebraic(XZ[inl])
                    if fitted is None:
                        break
                    center, radius = fitted
                d = np.linalg.norm(XZ - np.array([center]), axis=1)
                inlier_pct = float(np.mean(np.abs(d - radius) < 0.045))
                cx, cz = float(center[0]), float(center[1])
            ys = sparse[(sparse[:, 1] > -0.2) & (sparse[:, 1] < C.COLUMN_HEIGHT_M + 0.3), 1]
            if len(ys) > 20:
                y_lo = max(0.0, float(np.percentile(ys, 0.5)))
                y_hi = min(C.COLUMN_HEIGHT_M, float(np.percentile(ys, 99.5)))

        view_order = sorted(registered)
        if not view_order:
            return [], {"center": [cx, cz], "radius": radius, "inliers": inlier_pct}

        n_th, n_y = 448, 176
        th = np.linspace(-math.pi, math.pi, n_th, endpoint=False)
        ys = np.linspace(y_lo, y_hi, n_y)
        TH, YS = np.meshgrid(th, ys)
        P = np.stack([cx + radius * np.sin(TH), YS, cz + radius * np.cos(TH)],
                     axis=-1).reshape(-1, 3)
        normals = np.stack([np.sin(TH), np.zeros_like(TH), np.cos(TH)],
                           axis=-1).reshape(-1, 3)

        # photometric transfer: every surface sample takes its colour from the
        # most frontal registered view (weighted by cos(incidence))
        colours = np.zeros((P.shape[0], 3), dtype=np.float64)
        weights = np.zeros(P.shape[0], dtype=np.float64)
        for vid in view_order:
            pose = registered[vid]
            R, t = pose["R"], pose["t"]
            cam = (-R.T @ t).ravel()
            cam_len = float(np.linalg.norm(cam))
            if cam_len < 1e-6:
                continue
            cam_n = cam / cam_len
            w = np.clip(normals @ cam_n - math.cos(math.radians(62.0)), 0.0, None)
            if not np.any(w > 0):
                continue
            uv, z = self.views[vid].project(P)
            img = self._read_image(vid, gray=False)
            if img is None:
                continue
            H, W = img.shape[:2]
            ui = np.round(uv[:, 0]).astype(np.int64)
            vi = np.round(uv[:, 1]).astype(np.int64)
            ok = (z > 0.2) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (w > 0)
            if not np.any(ok):
                continue
            bgr = img[vi[ok], ui[ok]].astype(np.float64)
            colours[ok] += bgr[:, ::-1] * w[ok, None]
            weights[ok] += w[ok]

        have = weights > 1e-6
        rgb = np.zeros_like(colours)
        rgb[have] = colours[have] / weights[have, None]

        rows = []
        for k in np.flatnonzero(have):
            p = P[k]
            c = rgb[k]
            rows.append((p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2]), 0))
        meta = {"center": [cx, cz], "radius": radius, "inliers": inlier_pct,
                "samples": len(rows), "angular_coverage_deg": angular_coverage_deg}
        self._log(f"surface densification: cylinder centre ({cx:+.3f}, {cz:+.3f}) "
                  f"R = {radius:.4f} m ({inlier_pct * 100:.1f}% inliers), {len(rows)} samples")
        return rows, meta

    # ------------------------------------------------------------- export IO
    def write_ply(self, cloud: np.ndarray, path: Path) -> Path:
        """
        Binary little-endian PLY: x y z red green blue class, where ``class``
        is the defect class index so the front-end can colour-code hits.
        """
        path = Path(path)
        n = int(len(cloud))
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "comment AI 3D Structural Inspection System -- dense digital twin\n"
            f"comment object {C.COLUMN_LABEL}\n"
            f"comment generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property uchar class\n"
            "end_header\n").encode("ascii")
        verts = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                   ("class", "u1")])
        if n:
            verts["x"], verts["y"], verts["z"] = cloud[:, 0], cloud[:, 1], cloud[:, 2]
            for k, ch in enumerate(("red", "green", "blue")):
                verts[ch] = np.clip(cloud[:, 3 + k], 0, 255).astype(np.uint8)
            if cloud.shape[1] > 6:
                verts["class"] = np.clip(cloud[:, 6], 0, 255).astype(np.uint8)
        with path.open("wb") as fh:
            fh.write(header)
            fh.write(verts.tobytes())
        return path

    def write_obj(self, path: Path, n_th: int = 320, n_y: int = 140) -> Path:
        """
        Textured surface mesh of the reconstructed pier -- the solid "digital
        twin" -- with UVs mapped onto the unwrapped texture atlas, plus the
        internal reinforcement cage as a separate wireframe group.
        """
        path = Path(path)
        tex = C.ANALYSIS_DIR / "pier_texture_atlas.jpg"
        if not tex.exists():
            try:
                sim = _lazy_simulator()
                sim.save_texture_atlas(tex)
            except Exception:
                pass
        local_tex = path.with_name("digital_twin_texture.jpg")
        if tex.exists():
            shutil.copyfile(tex, local_tex)

        mtl = path.with_name("digital_twin.mtl")
        mtl.write_text(
            "# AI 3D Structural Inspection System -- digital twin material\n"
            "newmtl pier_concrete\nKa 0.20 0.20 0.20\nKd 1.00 1.00 1.00\n"
            "Ks 0.06 0.06 0.06\nNs 8.0\nd 1.0\nillum 2\n"
            f"map_Kd {local_tex.name}\n"
            "newmtl rebar_steel\nKa 0.16 0.14 0.12\nKd 0.42 0.38 0.34\n"
            "Ks 0.55 0.50 0.45\nNs 60.0\nd 1.0\nillum 2\n")

        lines = ["# AI 3D Structural Inspection System -- 3D digital twin",
                 f"# generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                 f"# object {C.COLUMN_LABEL}",
                 f"mtllib {mtl.name}", "o pier_surface", "usemtl pier_concrete"]
        R, H = C.COLUMN_RADIUS_M, C.COLUMN_HEIGHT_M
        th = np.linspace(0.0, 2.0 * math.pi, n_th + 1)
        ys = np.linspace(0.0, H, n_y + 1)
        for j, y in enumerate(ys):
            for i, t in enumerate(th):
                lines.append(f"v {R * math.sin(t):.5f} {y:.5f} {R * math.cos(t):.5f}")
        for j, y in enumerate(ys):
            for i, t in enumerate(th):
                lines.append(f"vt {i / n_th:.6f} {1.0 - j / n_y:.6f}")
        for j in range(n_y):
            for i in range(n_th):
                a = j * (n_th + 1) + i + 1
                b, c, d = a + 1, a + n_th + 1, a + n_th + 2
                lines.append(f"f {a}/{a} {b}/{b} {d}/{d} {c}/{c}")

        lines += ["o rebar_cage", "usemtl rebar_steel"]
        base = (n_th + 1) * (n_y + 1)
        r_sh = R - C.NOMINAL_COVER_MM / 1000.0 - 0.010
        count = 0
        for b in range(8):
            ang = 2.0 * math.pi * b / 8
            x, z = r_sh * math.sin(ang), r_sh * math.cos(ang)
            lines.append(f"v {x:.4f} 0.0000 {z:.4f}")
            lines.append(f"v {x:.4f} {H:.4f} {z:.4f}")
            lines.append(f"l {base + count + 1} {base + count + 2}")
            count += 2
        prev = None
        n_pt = 220
        for k in range(n_pt + 1):
            frac = k / n_pt
            ang = 2.0 * math.pi * frac * 20.0
            y = 0.05 + frac * (H - 0.10)
            lines.append(f"v {r_sh * math.sin(ang):.4f} {y:.4f} {r_sh * math.cos(ang):.4f}")
            if prev is not None:
                lines.append(f"l {prev} {base + count + k + 1}")
            prev = base + count + k + 1
        path.write_text("\n".join(lines) + "\n")
        return path

    # ---------------------------------------------------------------- misc
    def _sample_colour(self, view_index: int, xyz) -> List[int]:
        img = self._read_image(view_index, gray=False)
        if img is None:
            return [168, 168, 165]
        uv, z = self.views[view_index].project(np.asarray(xyz).reshape(1, 3))
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if z[0] <= 0 or not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
            return [168, 168, 165]
        b, g, r = img[int(round(v)), int(round(u))]
        return [int(r), int(g), int(b)]

    @staticmethod
    def _pair_matrix(pairs: List[PairGeometry]) -> List[List[int]]:
        n = C.N_VIEWS
        m = [[0] * n for _ in range(n)]
        for p in pairs:
            m[p.i][p.j] = p.inliers
            m[p.j][p.i] = p.inliers
        return m

    def _log(self, msg: str):
        line = f"[colmap {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        self.log.append(line)
        print(line, flush=True)


def _fit_circle_algebraic(XZ: np.ndarray):
    """
    Least-squares circle fit (Kasa): minimise sum((x^2+z^2) - 2 cx x - 2 cz z - k)^2
    which is linear in (cx, cz, k) with r = sqrt(k + cx^2 + cz^2).
    """
    XZ = np.asarray(XZ, dtype=np.float64).reshape(-1, 2)
    if len(XZ) < 3:
        return None
    A = np.column_stack([2.0 * XZ[:, 0], 2.0 * XZ[:, 1], np.ones(len(XZ))])
    b = (XZ ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cz, k = (float(v) for v in sol)
    r2 = k + cx * cx + cz * cz
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return np.array([cx, cz]), float(math.sqrt(r2))


def _project_with(pts, pose, K):
    """Project world points with an explicit (R, t) pose pair."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    cam = pts @ np.asarray(pose["R"]).T + np.asarray(pose["t"]).reshape(1, 3)
    z = cam[:, 2]
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * cam[:, 0] / zz + K[0, 2]
    v = K[1, 1] * cam[:, 1] / zz + K[1, 2]
    return np.stack([u, v], axis=1), z


def _coverage(pts: np.ndarray, shape) -> float:
    """Percentage of an 8x8 frame grid containing at least one keypoint."""
    if len(pts) == 0:
        return 0.0
    H, W = shape[:2]
    cells = np.zeros((8, 8), dtype=bool)
    ci = np.clip((pts[:, 0] / W * 8).astype(int), 0, 7)
    cj = np.clip((pts[:, 1] / H * 8).astype(int), 0, 7)
    cells[cj, ci] = True
    return float(cells.mean() * 100.0)


def _lazy_simulator():
    from .esp32_simulator import ESP32CamSimulator
    return ESP32CamSimulator(supersample=1)
